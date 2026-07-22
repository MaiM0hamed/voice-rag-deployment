# Electro Pi — Developer API Platform

## Authentication
The Electro Pi API uses bearer tokens. Tokens are generated in the portal under
Developer → API Keys. Each token is scoped to a single project and can be
revoked independently. Tokens do not expire automatically but are rotated
whenever a security incident is declared.

## Rate limits
The free tier allows **60 requests per minute** per token. The growth tier allows
600 requests per minute. Exceeding the limit returns HTTP 429 with a
`Retry-After` header. Sustained abuse results in temporary token suspension.

## Endpoints
The catalogue endpoint returns product metadata and live stock levels. The orders
endpoint supports creating and querying orders. Webhooks can be registered for
order status changes; webhook payloads are signed with HMAC-SHA256 and callers
must verify the signature before trusting the payload.

## Sandbox
A sandbox environment mirrors production with synthetic inventory. Sandbox tokens
are prefixed with `sk_test_` and never move real stock or charge real money.
