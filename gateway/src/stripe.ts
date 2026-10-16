// Stripe webhooks: verify signature, map price -> tier, upsert entitlements.
import Stripe from "stripe";
import { entitlementFor, type Env, type Tier } from "./auth";

function tierForPrice(priceId: string, mapJson: string | undefined): Tier {
  if (!mapJson) return "free";
  try {
    const map = JSON.parse(mapJson) as Record<string, Tier>;
    return map[priceId] ?? "free";
  } catch {
    return "free";
  }
}

export async function handleStripeWebhook(request: Request, env: Env): Promise<Response> {
  const signature = request.headers.get("stripe-signature");
  if (!signature) return new Response("missing signature", { status: 400 });

  const stripe = new Stripe(env.STRIPE_SECRET_KEY, { apiVersion: "2025-02-24.acacia" });
  const body = await request.text();

  let event: Stripe.Event;
  try {
    event = await stripe.webhooks.constructEventAsync(body, signature, env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    return new Response(`invalid signature: ${(err as Error).message}`, { status: 400 });
  }

  switch (event.type) {
    case "checkout.session.completed":
    case "customer.subscription.created":
    case "customer.subscription.updated": {
      const obj = event.data.object as Stripe.Checkout.Session | Stripe.Subscription;
      const customerId = typeof obj.customer === "string" ? obj.customer : obj.customer?.id;
      const subscription = "subscription" in obj && obj.subscription ? obj.subscription : obj;
      const priceId =
        typeof subscription === "object" && subscription && "items" in subscription
          ? subscription.items.data[0]?.price?.id
          : undefined;
      const tier = tierForPrice(priceId ?? "", (env as unknown as { PRICE_MAP?: string }).PRICE_MAP);
      if (customerId) {
        await env.USERS.put(
          `customer:${customerId}`,
          JSON.stringify(entitlementFor(customerId, tier)),
        );
      }
      break;
    }
    case "customer.subscription.deleted": {
      const sub = event.data.object as Stripe.Subscription;
      const customerId = typeof sub.customer === "string" ? sub.customer : sub.customer.id;
      await env.USERS.put(`customer:${customerId}`, JSON.stringify(entitlementFor(customerId, "free")));
      break;
    }
    default:
      break;
  }

  // idempotency: Stripe retries; handlers above are upserts so replays are safe.
  return new Response(JSON.stringify({ received: true }), {
    headers: { "content-type": "application/json" },
  });
}
