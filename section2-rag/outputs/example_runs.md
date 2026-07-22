# Section 2 — Example RAG Runs

_Generated 2026-07-22 17:28 by `run_examples.py`._

- Embedding backend: `huggingface`
- Generation model: `qwen2.5:1.5b` via Ollama
- Chunking: size 600, overlap 100
- Retrieval: top-4, hybrid=on, relevance threshold 0.5

---
### Q: How long do I have to return a development board, and when will I get my refund?

**Answer:** Development boards can be returned within 30 calendar days [1]. Refunds for development boards typically settle in **5 to 7 business days** via card refunds and **10 business days** via bank transfer refunds. Cash-on-delivery orders are refunded as store credit by default; customers may request a bank transfer instead.

**Citations:**
- [1] 02_returns_policy.md — Refund timing (chunk c94013115d9a)

<details><summary>Retrieved context</summary>

```
[1] 02_returns_policy.md — Refund timing
## Refund timing
Once a return is received and inspected, refunds are issued to the original
payment method. Card refunds settle in **5 to 7 business days**. Bank transfer
refunds settle in **10 business days**. Cash-on-delivery orders are refunded as
store credit by default; customers may request a bank transfer instead.

[2] 02_returns_policy.md — Return window
# Electro Pi — Returns and Refunds Policy  
## Return window
Standard components may be returned within **14 calendar days** of delivery.
Development boards and kits have an extended window of **30 calendar days**.
Custom-manufactured PCBs are **not returnable** unless they arrive faulty.

[3] 01_onboarding.md — Creating an account
# Electro Pi — Customer Onboarding Guide  
## Creating an account
New customers register at the Electro Pi portal using a company email address.
Registration requires a valid VAT number for business accounts. Personal
accounts are approved instantly; business accounts are reviewed manually and are
typically approved within **two business days**.

[4] 02_returns_policy.md — Condition requirements
## Condition requirements
Returned items must include the original anti-static packaging. Items showing
solder residue, physical modification, or firmware flashing are rejected by the
returns desk and shipped back at the customer's expense.
```

</details>

---

### Q: What are the shipping fees for an order under 1,500 EGP delivered in Cairo?

**Answer:** Orders below **1,500 EGP** ship free within Greater Cairo [2].

**Citations:**
- [2] 03_shipping.md — Coverage (chunk 7a694a7fd2d3)

<details><summary>Retrieved context</summary>

```
[1] 03_shipping.md — Shipping fees
## Shipping fees
Orders above **1,500 EGP** ship free within Greater Cairo. Below that threshold a
flat 60 EGP fee applies. Outside Greater Cairo the flat fee is 95 EGP and the
free-shipping threshold is 2,500 EGP.

[2] 03_shipping.md — Coverage
# Electro Pi — Shipping and Delivery  
## Coverage
Electro Pi ships to all governorates in Egypt. Greater Cairo and Alexandria are
served by same-day couriers for orders placed before 13:00. Other governorates
are served by Aramex and Bosta with a 2–4 business day transit time.

[3] 03_shipping.md — Delivery attempts
## Delivery attempts
Couriers make **two delivery attempts**. After a second failed attempt the parcel
returns to the Electro Pi warehouse and the customer is contacted to arrange
redelivery, which incurs the standard shipping fee.

[4] 01_onboarding.md — Support channels
## Support channels
Onboarding questions go to onboarding@electropi.example. The support hotline
operates Sunday to Thursday, 9:00–17:00 Cairo time. There is no weekend phone
support; tickets raised on Friday or Saturday are answered the next business day.
```

</details>

---

### Q: What happens if I exceed the API rate limit on the free tier?

**Answer:** Exceeding the limit returns HTTP 429 with a `Retry-After` header [1].

**Citations:**
- [1] 04_api_platform.md — Rate limits (chunk 124ed68582e8)

<details><summary>Retrieved context</summary>

```
[1] 04_api_platform.md — Rate limits
## Rate limits
The free tier allows **60 requests per minute** per token. The growth tier allows
600 requests per minute. Exceeding the limit returns HTTP 429 with a
`Retry-After` header. Sustained abuse results in temporary token suspension.

[2] 04_api_platform.md — Authentication
# Electro Pi — Developer API Platform  
## Authentication
The Electro Pi API uses bearer tokens. Tokens are generated in the portal under
Developer → API Keys. Each token is scoped to a single project and can be
revoked independently. Tokens do not expire automatically but are rotated
whenever a security incident is declared.

[3] 03_shipping.md — Shipping fees
## Shipping fees
Orders above **1,500 EGP** ship free within Greater Cairo. Below that threshold a
flat 60 EGP fee applies. Outside Greater Cairo the flat fee is 95 EGP and the
free-shipping threshold is 2,500 EGP.

[4] 01_onboarding.md — Verification tiers
## Verification tiers
Electro Pi uses three verification tiers:  
- **Tier 0 (unverified):** browse only, no purchases.
- **Tier 1 (email verified):** purchases up to 5,000 EGP per month.
- **Tier 2 (document verified):** unlimited purchases, requires a commercial
registration document and a national ID for the account owner.  
Upgrading from Tier 1 to Tier 2 requires uploading documents in the portal under
Settings → Verification. Review takes up to three business days.
```

</details>

---

### Q: What is Electro Pi's employee stock option vesting schedule?

**Answer:** I don't have enough information.

_Refused by the relevance gate; the LLM was not called._
