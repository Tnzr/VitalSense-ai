// Asculto API gateway (Cloudflare Worker + Hono).
//
// The gateway never runs inference: it authenticates, enforces quotas, issues
// presigned uploads, enqueues jobs, verifies Stripe webhooks, and proxies the
// synchronous single-head path to the Python inference worker.
import { Hono } from "hono";
import { AwsClient } from "aws4fetch";
import {
  authenticate,
  entitlementFor,
  incrementUsage,
  withinQuota,
  type Env as AuthEnv,
} from "./auth";
import { handleStripeWebhook } from "./stripe";

interface Env extends AuthEnv {
  AUDIO: R2Bucket;
  REPORTS: R2Bucket;
  JOBS: Queue;
  PRICE_MAP?: string;
}

type Bindings = Env;

const app = new Hono<{ Bindings: Bindings }>();

app.get("/health", (c) =>
  c.json({ status: "ok", service: "asculto-gateway", env: c.env.ENVIRONMENT }),
);

// --- presigned upload ---------------------------------------------------------
app.post("/v1/uploads/presign", async (c) => {
  const ent = await authenticate(c.req.raw, c.env);
  if (!ent) return c.json({ error: "unauthorized" }, 401);

  const { filename, contentType } = await c.req.json<{ filename?: string; contentType?: string }>();
  const key = `audio/${ent.userId}/${crypto.randomUUID()}-${(filename || "clip.wav").replace(/[^\w.\-]/g, "_")}`;

  const r2 = new AwsClient({
    accessKeyId: c.env.R2_ACCESS_KEY_ID,
    secretAccessKey: c.env.R2_SECRET_ACCESS_KEY,
    service: "s3",
    region: "auto",
  });
  const url = new URL(`https://${c.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/asculto-audio/${key}`);
  const signed = await r2.sign(
    new Request(url, { method: "PUT", headers: { "content-type": contentType || "audio/wav" } }),
    { aws: { signQuery: true } },
  );

  return c.json({ uploadUrl: signed.url, key, expiresInSeconds: 900 });
});

// --- synchronous analyze (macro / single head) --------------------------------
app.post("/v1/analyze", async (c) => {
  const ent = await authenticate(c.req.raw, c.env);
  if (!ent) return c.json({ error: "unauthorized" }, 401);
  if (!(await withinQuota(c.env, ent))) return c.json({ error: "quota exceeded", tier: ent.tier }, 402);

  const job = await parseJob(c.req.raw, c.env);
  if (!job) return c.json({ error: "provide multipart file, audio_b64 or audio_url" }, 400);

  const report = await callWorker(c.env, job);
  if (report.error) return c.json(report, 502);

  await incrementUsage(c.env, ent.userId, job.count ?? 1);
  return c.json(report);
});

// --- asynchronous report (queue -> worker -> signed result) -------------------
app.post("/v1/jobs", async (c) => {
  const ent = await authenticate(c.req.raw, c.env);
  if (!ent) return c.json({ error: "unauthorized" }, 401);
  if (!(await withinQuota(c.env, ent))) return c.json({ error: "quota exceeded", tier: ent.tier }, 402);

  const job = await parseJob(c.req.raw, c.env);
  if (!job) return c.json({ error: "provide multipart file, audio_b64 or audio_url" }, 400);

  const id = crypto.randomUUID();
  await c.env.JOBS.send({ ...job, id, userId: ent.userId });
  return c.json({ jobId: id, status: "queued", poll: `/v1/jobs/${id}` }, 202);
});

app.get("/v1/jobs/:id", async (c) => {
  const id = c.req.param("id");
  const obj = await c.env.REPORTS.get(`reports/${id}.json`);
  if (!obj) return c.json({ jobId: id, status: "pending" }, 202);
  const url = await presignGet(c.env, `reports/${id}.json`);
  return c.json({ jobId: id, status: "done", resultUrl: url });
});

// --- billing ------------------------------------------------------------------
app.post("/webhooks/stripe", (c) => handleStripeWebhook(c.req.raw, c.env));

// ----------------------------------------------------------------------------- helpers
async function parseJob(request: Request, env: Env): Promise<(Record<string, unknown> & { count?: number }) | null> {
  const contentType = request.headers.get("content-type") || "";
  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    const file = form.get("file") as unknown as File | string | null;
    const heads = String(form.get("heads") || "");
    if (!file || typeof file === "string") return null;
    const buf = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (const b of buf) binary += String.fromCharCode(b);
    return {
      audio_b64: btoa(binary),
      filename: file.name,
      count: 1,
      options: { heads: heads ? heads.split(",") : undefined, figures: String(form.get("figures") || "key") },
    };
  }
  const body = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  if (!body) return null;
  if (body.jobs && Array.isArray(body.jobs)) {
    return { jobs: body.jobs, count: body.jobs.length };
  }
  if (!body.audio_b64 && !body.audio_url && !body.key) return null;
  if (body.key) {
    const object = await env.AUDIO.get(String(body.key));
    if (!object) return null;
    const buf = new Uint8Array(await object.arrayBuffer());
    let binary = "";
    for (const b of buf) binary += String.fromCharCode(b);
    return { ...body, audio_b64: btoa(binary), filename: String(body.key).split("/").pop() };
  }
  return body;
}

async function callWorker(env: Env, job: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch(`${env.INFERENCE_URL}/run`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(job),
  });
  if (!res.ok) return { error: `worker ${res.status}` };
  return (await res.json()) as Record<string, unknown>;
}

async function presignGet(env: Env, key: string): Promise<string> {
  const r2 = new AwsClient({
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    service: "s3",
    region: "auto",
  });
  const url = new URL(`https://${env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com/asculto-reports/${key}`);
  const signed = await r2.sign(new Request(url, { method: "GET" }), { aws: { signQuery: true } });
  return signed.url;
}

export default {
  fetch: app.fetch,
  async queue(batch: MessageBatch<Record<string, unknown> & { id: string }>, env: Env): Promise<void> {
    for (const msg of batch.messages) {
      try {
        const report = await callWorker(env, msg.body);
        await env.REPORTS.put(`reports/${msg.body.id}.json`, JSON.stringify(report), {
          httpMetadata: { contentType: "application/json" },
        });
        await incrementUsage(env, String(msg.body.userId || "anon"), Number(msg.body.count ?? 1));
        msg.ack();
      } catch (err) {
        console.error("job failed", msg.body.id, err);
        msg.retry();
      }
    }
  },
};

// keep the helper referenced for future per-key entitlement lookups
export const _entitlementFor = entitlementFor;
