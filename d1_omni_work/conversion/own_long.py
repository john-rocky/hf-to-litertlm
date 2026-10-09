"""`own_long_3400`: one long invented state (an incident postmortem draft of an invented invoicing SaaS, a timeline log
and an attached metrics JSON) sized to ~3,400 state tokens under the d1-omni tokenizer, and three questions (noul /
choice / score) whose answers the text decides.

Every person, organisation, product, service and place name is invented; there is no real URL, address, trademark or
date-bound event in it. Routine log lines (`FILLER`) pad the timeline to the size; `N_FILLER` was tuned once with
encode() and is fixed here (fx_define.py asserts the measured state length stays in 3,300..3,500).
"""
import json

N_FILLER = 28  # tuned 2026-10-08 with fx_define.py --tune: 28 -> 3,403 state tokens (24 -> 3,290, 32 -> 3,525)

HEADER = """Tallowmere Invoicing - incident review draft INC-0412 (internal, not for customers yet)
Prepared by: Bayo Adeyemi-Stroud (incident commander). Contributors: Ines Varga-Holm (on-call), Lucie Ferrandis-Kemp (payments), Mirela Tanaka-Booth (databases), Hamish Okonkwo-Reyes (support lead).
Status: mitigated and closed; refunds complete; customer notice drafted.

Summary
Between 10:02 and 10:47 UTC the checkout flow of Tallowmere Invoicing submitted a second card authorization for some orders after a gateway timeout. The payment router retried the capture after the idempotency key had already expired, so the card network treated the retry as a new payment. Checkout stayed available the whole time (availability above 99.6%), but checkout latency rose and a subset of customers saw two identical charges on their statements. All duplicate charges were refunded by 11:40 UTC. The sections below give the timeline as it was logged, the customer tickets that reached support first, the metrics attachment, and the follow-up actions.

Timeline (UTC, from the incident channel and the service logs)
"""

# (time, component, level, message): the incident as it unfolded, including the two signals that turned out to be
# unrelated (a database disk warning and a certificate reminder) and the card-network latency spike that was ruled out.
CORE = [
    ("09:52:10", "deploy-pipeline", "INFO", "change CR-2207 approved by Lucie Ferrandis-Kemp: pay-router retry policy update (ticket PAY-883)"),
    ("09:58:41", "deploy-pipeline", "INFO", "CR-2207 queued for staged rollout: region-north, then region-south, then region-east, two minutes apart"),
    ("09:58:42", "deploy-pipeline", "INFO", "CR-2207 diff: idempotency_window_s 300 -> 30; max_capture_retries 2 -> 4; retry_backoff_s [5, 15] -> [5, 15, 30, 45]"),
    ("10:02:03", "pay-router", "INFO", "config CR-2207 applied in region-north (version 118 -> 119)"),
    ("10:04:05", "pay-router", "INFO", "config CR-2207 applied in region-south (version 118 -> 119)"),
    ("10:06:02", "pay-router", "INFO", "config CR-2207 applied in region-east (version 118 -> 119)"),
    ("10:08:57", "card-adapter", "WARN", "card network authorization p95 latency 2.4 s over 60 s (normal 0.8 s); 14 gateway timeouts"),
    ("10:09:12", "pay-router", "WARN", "order T-55120: capture timed out after 10 s; retry 1 scheduled in 5 s"),
    ("10:09:44", "pay-router", "WARN", "order T-55120: retry 3 after 30 s backoff; idempotency key k-55120 expired (window 30 s); new key issued"),
    ("10:09:45", "card-adapter", "INFO", "order T-55120: authorization approved (second authorization for this order)"),
    ("10:11:30", "card-adapter", "INFO", "card network authorization p95 latency 1.9 s over 60 s; 9 gateway timeouts"),
    ("10:13:02", "ledger-api", "WARN", "duplicate capture check: order T-55120 has 2 settled captures of 84.00; flagged for review"),
    ("10:14:40", "ledger-api", "WARN", "duplicate capture check: 11 orders with 2 settled captures in the last 5 minutes"),
    ("10:15:20", "card-adapter", "INFO", "card network authorization p95 latency 0.9 s over 60 s; 0 gateway timeouts"),
    ("10:16:05", "support-desk", "INFO", "ticket S-20931 opened: customer reports being charged twice for one invoice"),
    ("10:17:48", "support-desk", "INFO", "ticket S-20934 opened: customer reports two identical card charges a minute apart"),
    ("10:19:10", "ledger-api", "WARN", "duplicate capture check: 38 orders with 2 settled captures in the last 10 minutes"),
    ("10:21:02", "metrics", "ERROR", "alert DuplicateCaptureRate fired (severity page): 0.9% of captures duplicated over 10 minutes"),
    ("10:21:40", "metrics", "INFO", "page sent to on-call Ines Varga-Holm"),
    ("10:22:31", "metrics", "INFO", "page acknowledged by Ines Varga-Holm"),
    ("10:24:15", "incident-bot", "INFO", "incident INC-0412 declared, commander Bayo Adeyemi-Stroud, channel opened"),
    ("10:26:02", "incident-bot", "INFO", "hypothesis 1 (Ines): card network latency spike caused double authorizations upstream"),
    ("10:27:50", "card-adapter", "INFO", "card network status page: all systems normal; our p95 back to 0.8 s since 10:15"),
    ("10:28:33", "ledger-api", "WARN", "duplicate capture check: 19 more duplicated orders since 10:19 although network latency is normal"),
    ("10:29:05", "incident-bot", "INFO", "hypothesis 1 rejected: duplicates continue after latency recovered"),
    ("10:31:14", "db-replica-2", "WARN", "disk usage 88% on the data volume (threshold 90%); archive job running normally"),
    ("10:31:40", "incident-bot", "INFO", "Mirela: replica disk warning is routine weekly growth, not related; no write errors"),
    ("10:33:18", "incident-bot", "INFO", "hypothesis 2 (Lucie): CR-2207 made the idempotency window (30 s) shorter than the longest retry backoff (45 s)"),
    ("10:34:52", "pay-router", "WARN", "order T-55871: retry 4 after 45 s backoff; idempotency key expired; new key issued"),
    ("10:35:30", "incident-bot", "INFO", "Lucie confirms: every duplicated order shows a retry after the key expired; decision to roll back CR-2207"),
    ("10:38:06", "deploy-pipeline", "INFO", "rollback of CR-2207 started: region-north, region-south, region-east"),
    ("10:41:09", "pay-router", "INFO", "config rolled back in region-north (version 119 -> 120 = 118 settings)"),
    ("10:43:11", "pay-router", "INFO", "config rolled back in region-south (version 119 -> 120 = 118 settings)"),
    ("10:45:04", "pay-router", "INFO", "config rolled back in region-east (version 119 -> 120 = 118 settings)"),
    ("10:47:30", "ledger-api", "INFO", "duplicate capture check: 0 new duplicated orders in the last 5 minutes"),
    ("10:52:00", "incident-bot", "INFO", "plan: refund the second capture of every flagged order; Hamish prepares the customer notice"),
    ("10:58:20", "ledger-api", "INFO", "duplicate capture report: 212 orders from 212 distinct customers have two settled captures"),
    ("11:05:12", "refund-job", "INFO", "refund batch 1 submitted: 120 orders, total 9,846.50"),
    ("11:10:03", "status-page", "INFO", "reminder: the status page certificate expires in 21 days; renewal ticket OPS-1290 already scheduled"),
    ("11:12:44", "refund-job", "INFO", "refund batch 1 confirmed by the card network: 120/120"),
    ("11:20:31", "refund-job", "INFO", "refund batch 2 submitted: 92 orders, total 7,112.25"),
    ("11:31:58", "refund-job", "INFO", "refund batch 2 confirmed by the card network: 92/92"),
    ("11:40:00", "incident-bot", "INFO", "all 212 duplicate captures refunded; support macro updated for tickets about double charges"),
    ("11:45:16", "metrics", "INFO", "alert DuplicateCaptureRate resolved"),
    ("12:00:00", "incident-bot", "INFO", "incident INC-0412 closed by Bayo Adeyemi-Stroud; review meeting booked for the next working day"),
]

# Routine lines: the same few health and traffic messages the services print every couple of minutes.
_ROUTINE = [
    ("edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency {a} ms"),
    ("ledger-api", "INFO", "GET /invoices served {b} requests in the last minute, error rate 0.0%"),
    ("metrics", "INFO", "scrape ok: 214 targets, 0 stale"),
    ("queue-relay", "INFO", "topic invoice.events depth {c}, consumers 4, lag 0.3 s"),
    ("checkout-web", "INFO", "checkout page p95 render {d} ms, availability 100.0% over 60 s"),
]


def filler(n):
    """n routine lines spread over 09:50..11:59 (deterministic, no randomness)."""
    out = []
    for i in range(n):
        minute = 50 + (i * 130) // max(1, n)          # 09:50 .. 11:59
        hh, mm = 9 + minute // 60, minute % 60
        comp, level, msg = _ROUTINE[i % len(_ROUTINE)]
        out.append((f"{hh:02d}:{mm:02d}:{(17 * i) % 60:02d}", comp, level,
                    msg.format(a=38 + (i * 7) % 9, b=280 + (i * 13) % 50, c=30 + (i * 11) % 25, d=410 + (i * 37) % 300)))
    return out


TICKETS = """
First customer tickets (verbatim excerpts, names removed)
- S-20931, 10:16: "I paid invoice 4471 this morning and my card shows the same amount twice, 84.00 each. Please fix this, I only bought it once."
- S-20934, 10:17: "Two identical charges a minute apart from Tallowmere. Is this a mistake or was I billed for two subscriptions?"
- S-20940, 10:23: "My shop's card was charged twice for the monthly plan. Not urgent, but I want the extra charge back."
- S-20951, 10:36: "Double charge again, same as my colleague. Is checkout broken? It took ages to load as well."
- S-20966, 11:02: "Saw the duplicate charge disappear from pending, thanks. Will the refund show up as a separate line?"
"""

ACTIONS = """
Follow-up actions
1. Make the idempotency window at least twice the sum of all retry backoffs, and reject a config that breaks this rule in the deploy pipeline (owner: Lucie Ferrandis-Kemp).
2. Add the duplicate capture check to the canary stage so a staged rollout halts before the second region (owner: Bayo Adeyemi-Stroud).
3. Send the customer notice for the 212 affected customers with the refund reference (owner: Hamish Okonkwo-Reyes).
4. No action for the replica disk warning or the certificate reminder; both were tracked before the incident.
"""

ATTACHMENT = {
    "incident": "INC-0412",
    "window_utc": {"start": "10:02", "mitigated": "10:47", "closed": "12:00"},
    "change": {"id": "CR-2207", "component": "pay-router",
               "before": {"idempotency_window_s": 300, "max_capture_retries": 2, "retry_backoff_s": [5, 15]},
               "after": {"idempotency_window_s": 30, "max_capture_retries": 4, "retry_backoff_s": [5, 15, 30, 45]},
               "rolled_back": True},
    "regions": [
        {"name": "region-north", "orders": 4120, "duplicate_captures": 97, "refunded": 97, "checkout_p95_ms_peak": 3120,
         "checkout_availability_pct": 99.7},
        {"name": "region-south", "orders": 3988, "duplicate_captures": 71, "refunded": 71, "checkout_p95_ms_peak": 2840,
         "checkout_availability_pct": 99.6},
        {"name": "region-east", "orders": 3765, "duplicate_captures": 44, "refunded": 44, "checkout_p95_ms_peak": 2510,
         "checkout_availability_pct": 99.8},
    ],
    "totals": {"orders": 11873, "duplicate_captures": 212, "customers_affected": 212, "refunded": 212,
               "refund_amount": 16958.75, "currency": "EUR"},
    "ruled_out": [
        {"signal": "card network latency spike 10:08-10:15", "reason": "duplicates continued after latency recovered"},
        {"signal": "db-replica-2 disk usage 88%", "reason": "routine growth, below threshold, no write errors"},
        {"signal": "status page certificate expiry in 21 days", "reason": "renewal already scheduled, unrelated"},
    ],
}

QUESTIONS = {
    "double_charge": {"type": "noul", "instructions": "Were any customers charged twice because of this incident?",
                      "criteria": {"true": "At least one customer was charged twice",
                                   "false": "No customer was charged twice"}},
    "root_cause": {"type": "choice", "instructions": "What was the root cause of the incident?",
                   "criteria": {"config_change": "A configuration change to the payment router",
                                "disk_full": "A database volume running out of space",
                                "certificate": "An expired TLS certificate",
                                "network_outage": "An outage at the card network"}},
    "impact": {"type": "score", "instructions": "How severe was the customer impact?",
               "criteria": ["No customer impact", "Some customers affected",
                            "Most customers unable to use the service"]},
}
GOLD = {"double_charge": "true", "root_cause": "config_change", "impact": "1"}


def state(n_filler=N_FILLER):
    lines = sorted(CORE + filler(n_filler), key=lambda r: r[0])
    log = "\n".join(f"{t} {comp} {level} {msg}" for t, comp, level, msg in lines)
    return (HEADER + log + "\n" + TICKETS + ACTIONS + "\nAttachment metrics.json\n"
            + json.dumps(ATTACHMENT, indent=2, ensure_ascii=False))
