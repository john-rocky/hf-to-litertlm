"""Own fixture records (`own_*`): written for this port. Every person, organisation, product and place name is invented;
no real person, company, trademark, address or URL appears. Each record is a /v1/systemone request plus the answer the
author of the record intended (`gold`, keys as kev.api.question_keys reports them).

Coverage: support ticket / incident report / email thread / meeting notes / product review / JSON order, invoice and
sensor readings; noul, choice and score mixed (>= 6 questions each); choice with 2..10 options (one with 10); score
with 3..5 levels; noul with criteria (>= 2) and one question without instructions; three long states (~1,500 tokens: a
service log, a contract excerpt, board minutes) whose rows (state + branch) are between 1,024 and 2,048 tokens; one
request of five short questions about a ~100-token state (~270 tokens in all)."""

LOG_LINES = [
    ("01:00:02", "edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency 41 ms"),
    ("01:00:05", "auth-gateway", "INFO", "token refresh batch complete: 1,284 sessions renewed, 0 failures"),
    ("01:02:11", "billing-api", "INFO", "GET /invoices served 312 requests in the last minute, error rate 0.0%"),
    ("01:04:40", "db-primary", "INFO", "checkpoint complete: wrote 18,204 buffers, sync 1.2 s"),
    ("01:05:00", "metrics", "INFO", "scrape ok: 214 targets, 0 stale"),
    ("01:07:19", "queue-relay", "INFO", "topic billing.events depth 42, consumers 4, lag 0.3 s"),
    ("01:10:02", "edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency 39 ms"),
    ("01:12:33", "db-replica-2", "INFO", "replication lag 0.4 s, apply rate 2,100 records/s"),
    ("01:15:08", "billing-worker", "INFO", "processed 518 invoice events, 0 retries"),
    ("01:18:51", "db-replica-1", "INFO", "replication lag 0.3 s, apply rate 2,080 records/s"),
    ("01:20:02", "edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency 43 ms"),
    ("01:22:14", "db-replica-2", "WARN", "disk usage 91% on /var/lib/db (threshold 90%); write-ahead log archive backlog 3.1 GB"),
    ("01:24:30", "auth-gateway", "INFO", "login rate 22/s, rejected 0.4% (bad password), no upstream errors"),
    ("01:26:45", "metrics", "WARN", "alert DiskUsageHigh fired for db-replica-2 (severity warning)"),
    ("01:28:02", "billing-api", "INFO", "GET /invoices served 298 requests in the last minute, error rate 0.0%"),
    ("01:31:09", "db-replica-2", "WARN", "disk usage 96% on /var/lib/db; archive job could not delete segments still needed by a stalled backup"),
    ("01:33:40", "db-replica-1", "INFO", "replication lag 0.3 s, apply rate 2,060 records/s"),
    ("01:35:12", "billing-worker", "INFO", "processed 487 invoice events, 0 retries"),
    ("01:38:27", "db-replica-2", "ERROR", "no space left on device while writing WAL segment; replication apply stopped"),
    ("01:38:29", "db-replica-2", "ERROR", "replica marked unhealthy by its own watchdog; still accepting read connections"),
    ("01:39:55", "billing-api", "WARN", "read pool: 3 slow queries on db-replica-2 (over 2,000 ms)"),
    ("01:41:16", "billing-api", "ERROR", "read timeout on db-replica-2 after 5,000 ms; GET /invoices returned 503 to 37 requests in the last minute"),
    ("01:42:03", "edge-lb", "WARN", "upstream billing-api error rate 11.8% over 60 s"),
    ("01:42:30", "auth-gateway", "INFO", "login rate 21/s, rejected 0.5% (bad password), no upstream errors"),
    ("01:44:18", "queue-relay", "WARN", "topic billing.events depth 1,906 and rising; consumer billing-worker slow (waiting on reads)"),
    ("01:45:02", "billing-worker", "WARN", "processed 61 invoice events, 140 retries (read timeout)"),
    ("01:46:40", "billing-api", "ERROR", "GET /invoices returned 503 to 52 requests in the last minute"),
    ("01:47:05", "metrics", "ERROR", "alert BillingErrorRate fired (severity page); on-call engineer Rhea Castellane paged"),
    ("01:49:31", "metrics", "INFO", "page acknowledged by Rhea Castellane"),
    ("01:51:12", "db-primary", "INFO", "primary healthy: commits 1,450/s, no replication slot errors on db-replica-1"),
    ("01:53:44", "billing-api", "INFO", "operator change: read pool weight for db-replica-2 set to 0, db-replica-1 set to 100"),
    ("01:55:09", "billing-api", "INFO", "read pool drained from db-replica-2; all reads now served by db-replica-1"),
    ("01:57:20", "billing-api", "WARN", "GET /invoices returned 503 to 6 requests in the last minute (in-flight requests on the old pool)"),
    ("02:00:03", "edge-lb", "INFO", "upstream billing-api error rate 0.9% over 60 s"),
    ("02:03:11", "billing-api", "INFO", "GET /invoices served 305 requests in the last minute, error rate 0.0%"),
    ("02:04:47", "queue-relay", "INFO", "topic billing.events depth 1,212 and falling; consumers 4"),
    ("02:06:58", "billing-worker", "INFO", "processed 702 invoice events, 3 retries"),
    ("02:08:30", "db-replica-2", "INFO", "operator action: stalled backup cancelled, 40 GB volume extension attached"),
    ("02:10:14", "db-replica-2", "INFO", "disk usage 58% on /var/lib/db; replication apply resumed, lag 1,840 s"),
    ("02:14:02", "queue-relay", "INFO", "topic billing.events depth 220 and falling; consumers 4"),
    ("02:16:40", "db-replica-2", "INFO", "replication lag 610 s, apply rate 6,900 records/s (catching up)"),
    ("02:20:05", "metrics", "INFO", "alert BillingErrorRate resolved"),
    ("02:22:27", "db-replica-2", "INFO", "replication lag 95 s, apply rate 5,400 records/s"),
    ("02:26:13", "queue-relay", "INFO", "topic billing.events depth 38, consumers 4, lag 0.2 s"),
    ("02:31:36", "db-replica-2", "INFO", "replication lag 0.5 s; watchdog marked replica healthy"),
    ("02:33:02", "billing-api", "INFO", "operator change: read pool weight for db-replica-2 restored to 50, db-replica-1 to 50"),
    ("02:35:44", "metrics", "INFO", "alert DiskUsageHigh resolved for db-replica-2"),
    ("02:40:02", "edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency 40 ms"),
    ("02:42:19", "billing-api", "INFO", "GET /invoices served 289 requests in the last minute, error rate 0.0%"),
    ("02:45:51", "billing-worker", "INFO", "processed 455 invoice events, 0 retries"),
    ("02:50:02", "edge-lb", "INFO", "health check ok: 6/6 upstreams healthy, p50 latency 38 ms"),
    ("02:52:30", "auth-gateway", "INFO", "token refresh batch complete: 1,301 sessions renewed, 0 failures"),
    ("02:55:16", "db-replica-1", "INFO", "replication lag 0.3 s, apply rate 2,040 records/s"),
    ("02:57:48", "db-replica-2", "INFO", "replication lag 0.4 s, apply rate 2,090 records/s"),
    ("03:00:02", "metrics", "INFO", "scrape ok: 214 targets, 0 stale; no active alerts"),
]

LOG_STATE = ("Ombrevale Hosting, production cluster east-2. Log excerpt for the night of 28 September 2026 (times UTC). "
             "Fields: time, service, level, message.\n" + "\n".join(f"{t} {svc} {lvl} {msg}" for t, svc, lvl, msg in LOG_LINES))

CONTRACT_STATE = """MASTER SERVICES AGREEMENT (excerpt)

This Master Services Agreement is made between Corvane Analytics Ltd. ("Provider") and Brackenford Pottery Ltd. ("Client"), each a "Party".

1. Definitions
1.1 "Services" means the demand-forecasting and stock-planning services described in Schedule A, including the hosted dashboard, the weekly forecast report and the support desk.
1.2 "Fees" means the amounts payable by the Client under clause 4.
1.3 "Business Day" means any day other than a Saturday, a Sunday or a public holiday where the Client has its registered office.
1.4 "Client Data" means all sales, stock and supplier data that the Client uploads to the Services or that the Provider receives from the Client for the purpose of the Services.

2. Services
2.1 The Provider shall provide the Services with reasonable skill and care and in accordance with Schedule A.
2.2 The Provider shall make the hosted dashboard available at least 99.5% of each calendar month, excluding planned maintenance announced at least five Business Days in advance.
2.3 If availability falls below 99.5% in any month, the Client is entitled to a service credit of 5% of that month's Fees for each full percentage point of shortfall, up to a maximum of 25% of that month's Fees. Service credits are the Client's sole remedy for unavailability.

3. Term and Renewal
3.1 This Agreement starts on 1 October 2026 and continues for an initial term of twenty-four (24) months.
3.2 After the initial term, this Agreement renews automatically for successive renewal terms of twelve (12) months each, unless either Party gives written notice of non-renewal at least sixty (60) days before the end of the then-current term.

4. Fees and Payment
4.1 The Client shall pay a monthly fee of 2,400.00 for the Services, plus a one-time onboarding fee of 3,000.00 payable with the first invoice.
4.2 The Provider shall invoice monthly in advance. Each invoice is due within thirty (30) days of its date.
4.3 Overdue amounts carry interest at 1% per month from the due date until payment.
4.4 The Provider may increase the monthly fee once in each twelve-month period, by no more than 4%, on sixty (60) days' written notice.

5. Termination
5.1 The Client may terminate this Agreement for convenience at any time by giving the Provider not less than ninety (90) days' written notice. Fees paid in advance for any period after the termination date will be refunded.
5.2 Either Party may terminate this Agreement with immediate effect by written notice if the other Party commits a material breach and fails to remedy it within thirty (30) days after receiving written notice describing the breach.
5.3 The Provider may suspend the Services if any undisputed invoice remains unpaid for more than forty-five (45) days after its due date, provided that it has given the Client at least ten (10) Business Days' written warning.
5.4 On termination, the Provider shall make the Client Data available for export for sixty (60) days and shall then delete it, unless the law requires it to be kept.

6. Confidentiality
6.1 Each Party shall keep the other Party's confidential information secret and use it only to perform or receive the Services.
6.2 This clause does not apply to information that is or becomes public through no fault of the receiving Party, or that the receiving Party already lawfully held.
6.3 The obligations in this clause continue for three (3) years after this Agreement ends.

7. Data Protection
7.1 The Provider shall process Client Data only on the Client's documented instructions and only for the purpose of the Services.
7.2 The Provider shall notify the Client without undue delay, and in any event within forty-eight (48) hours, after becoming aware of any breach of security affecting Client Data.
7.3 The Provider shall not move Client Data to a subcontractor without the Client's prior written consent, which shall not be unreasonably withheld.

8. Liability
8.1 Nothing in this Agreement limits liability for fraud, or for death or personal injury caused by negligence.
8.2 Subject to clause 8.1, each Party's total liability arising out of or in connection with this Agreement in any twelve-month period is limited to the total Fees paid or payable by the Client in that twelve-month period.
8.3 Neither Party is liable for loss of profits, loss of business or any indirect or consequential loss.

9. Insurance
9.1 The Provider shall maintain professional indemnity insurance with a limit of not less than 500,000 per claim for the term of this Agreement and for one year after it ends.

10. Governing Law
10.1 This Agreement is governed by the laws of the jurisdiction in which the Client has its registered office, and the courts of that jurisdiction have exclusive jurisdiction over any dispute.

11. Notices
11.1 Notices under this Agreement must be in writing and delivered by hand or by recorded post to the address of the receiving Party set out in Schedule B, with a copy by email to the contact named there.
11.2 A notice is deemed received on the second Business Day after posting, or on delivery if delivered by hand.

12. Assignment and Subcontracting
12.1 Neither Party may assign or transfer this Agreement without the prior written consent of the other Party, except that either Party may assign it to a successor to all or substantially all of its business on written notice.
12.2 The Provider may use subcontractors to deliver parts of the Services, subject to clause 7.3, and remains responsible for their acts and omissions as if they were its own.

13. Force Majeure
13.1 Neither Party is liable for any delay or failure to perform caused by events beyond its reasonable control, including fire, flood, war, epidemic, or the failure of a public utility, provided that it notifies the other Party promptly and uses reasonable efforts to resume performance.
13.2 This clause does not excuse the Client's obligation to pay Fees for Services already performed.

14. Entire Agreement and Changes
14.1 This Agreement, together with its Schedules, is the entire agreement between the Parties about its subject matter and replaces any earlier proposals or understandings.
14.2 Any change to this Agreement must be agreed in writing and signed by an authorised representative of each Party.
14.3 If any provision of this Agreement is found invalid or unenforceable, the remaining provisions continue in full force."""

MINUTES_STATE = """Saltmere Growers Cooperative: minutes of the board meeting held on 2 September 2026 in the packing shed office.

Present: Wenna Thorley (chair), Idris Pennock (treasurer), Marisol Agbaje (secretary), Corin Hale, Bettina Ruskin, Joachim Amberley, Fen Okafor.
In attendance: Dorran Vessey (site manager, items 4 to 7).
Apologies: none.

1. Opening
The chair opened the meeting at 18:05 and confirmed that seven of the seven board members were present, so the meeting was quorate.

2. Minutes of the previous meeting
The minutes of the meeting held on 5 August 2026 were approved without changes. Proposed by Corin Hale, seconded by Fen Okafor.

3. Treasurer's report
Idris Pennock reported that sales for July and August were 6% above the same months last year, mainly because of the early tomato harvest. Operating costs rose by 9%, driven by electricity for the greenhouse fans during the August heatwave. The cash reserve stands at 41,300, which is above the board's minimum of 30,000. The treasurer asked members to submit expense claims within thirty days, as several claims from June arrived late. Fen Okafor asked how much of the electricity increase was a one-off. The treasurer said that about two thirds came from the heatwave weeks and the rest from the new tariff that started in July, which will continue. He proposed that the board review the tariff at the November meeting, when the supplier's winter prices are published. Bettina Ruskin asked whether the cooperative could claim the rural energy rebate again this year; the treasurer will check the deadline and report back. The report was accepted.

4. Irrigation upgrade budget
Dorran Vessey presented the proposal to replace the drip lines and controllers in greenhouses 1 to 3, with a budget of 38,000. The site manager explained that the current controllers fail about twice a month and that spare parts are no longer produced. One quote has been received, from a supplier who could install in November. Joachim Amberley said the board should not approve a sum this large on a single quote, and Bettina Ruskin agreed. The treasurer added that the cash reserve could cover the cost but would fall close to the minimum. After discussion, the chair proposed that the vote be postponed until two further quotes are available. The board agreed to defer the decision to the October meeting. Dorran Vessey will request the two additional quotes by 20 September.

5. Greenhouse 4 lease renewal
The lease on greenhouse 4 expires on 31 December 2026. The landowner has offered a renewal for five years at the current rent plus 3% per year. Fen Okafor noted that greenhouse 4 produces about a fifth of the cooperative's cucumbers and that moving production elsewhere would cost more than the rent increase. Corin Hale voted against renewal, arguing that the cooperative should wait for the outcome of the irrigation decision before committing to five more years. The motion to renew the lease was carried by six votes to one. The secretary will sign the renewal on behalf of the board.

6. Packing shed safety audit
The site manager reported on the safety audit carried out on 21 August. Two findings were rated high: a missing guard on the conveyor belt at the grading station and an emergency exit partly blocked by stacked crates. Both were fixed the same week. Four findings were rated low, including faded floor markings and an out-of-date first aid poster; these will be fixed by the end of September. The auditor also recommended a written procedure for cleaning the grading machine, which the site manager will draft with the two senior packers. Corin Hale asked whether the audit had looked at the cold room; the site manager confirmed that the cold room was inspected and had no findings. The board thanked the site manager and asked for a short update at the next meeting.

7. Seasonal staffing
Marisol Agbaje reported that eleven seasonal pickers have been hired for September and October, two fewer than planned. The site manager will move two members of the packing team to picking on weekdays until the end of the harvest. Idris Pennock noted that the staffing budget for the season is 18,500 and that spending to the end of August was 9,960, so the budget can absorb two weeks of overtime if the harvest runs late. The board agreed that seasonal pay will stay at the August rate.

8. Farmers' market stall
Bettina Ruskin proposed running a stall at the Saturday market in the town square from October to December, staffed by volunteer members. The cost is 45 per Saturday. Approved unanimously. Bettina Ruskin will draw up a volunteer rota.

9. Packaging supplier
Marisol Agbaje reported that the cooperative's current supplier of cardboard trays will raise prices by 7% from November. She has received a quote from a second supplier that is 3% cheaper than the current price but requires a minimum order of 20,000 trays. The site manager said the packing shed has space for that quantity only if the old crates in the back store are cleared. The board asked the secretary to ask both suppliers for samples and to bring a recommendation to the October meeting. No decision was taken.

10. Member communications
Fen Okafor proposed sending members a short monthly newsletter by post and by text message, covering prices, harvest dates and board decisions. Members who attended the summer open day had said that they often hear about decisions late. The board agreed to try the newsletter for three months, starting in October, and to review it in January. Fen Okafor will prepare the first issue with the secretary.

11. Any other business
Joachim Amberley asked whether the cooperative could share a delivery van with a neighbouring orchard to cut transport costs. The chair asked him to bring figures to the October meeting. Corin Hale reminded the board that the annual general meeting must be announced at least six weeks in advance; the secretary will propose dates at the next meeting.

12. Next meeting
The next board meeting will be held on 7 October 2026 at 18:00 in the packing shed office. The chair closed the meeting at 19:52."""


OWN = [
    {
        "id": "own_ticket_01", "kind": "support ticket",
        "request": {
            "state": "Ticket #48213, opened by Mara Quellen.\n\nHi, I ordered the Thistlebeam desk lamp (order TB-20931) on September 14 and was charged twice on my card: two identical charges of 64.90 appeared on September 15. The lamp arrived yesterday and works perfectly, no complaints there. Could you refund the duplicate charge? I need it sorted before my card statement closes on Friday. No rush on anything else, and thanks for the lovely lamp.",
            "questions": {
                "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                         "criteria": {"billing": "Charges, refunds and invoices", "shipping": "Deliveries, tracking and lost parcels",
                                      "returns": "Exchanges and sending a product back", "technical": "Product faults and setup help"}},
                "deadline": {"type": "noul", "instructions": "Does the customer ask for action by a specific deadline?",
                             "criteria": {"true": "The ticket names a day or date by which something must happen", "false": "No deadline is stated"}},
                "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["Calm", "Annoyed", "Angry"]},
            },
        },
        "gold": {"team": "billing", "deadline": "true", "mood": "0"},
    },
    {
        "id": "own_incident_02", "kind": "incident report",
        "request": {
            "state": "Incident report IR-1177, Orrinwick Foods cold-storage site B.\n\nAt 02:14 the compressor in freezer room 3 stopped after its breaker tripped. The on-call technician, Tobias Ferrow, reset the breaker at 02:51 and the compressor restarted. The room temperature rose from -21 C to -9 C during the outage. Product inspection found that 40 cartons of frozen berries had partly thawed; they were discarded. No staff were injured. The same breaker had tripped twice in the previous month. An electrician is booked for Monday.",
            "questions": {
                "severity": {"type": "score", "instructions": "How severe was this incident?",
                             "criteria": ["No impact", "Minor: no product lost", "Moderate: some product lost, nobody hurt",
                                          "Major: most stock lost or someone slightly hurt", "Critical: serious injury or the site shut down"]},
                "cause": {"type": "choice", "instructions": "What was the immediate cause of the outage?",
                          "criteria": {"power": "Electrical supply or breaker problem", "mechanical": "Mechanical failure inside the compressor",
                                       "human": "A person's mistake", "weather": "Outside weather conditions", "unknown": "The report does not say"}},
                "repeat": {"type": "noul", "instructions": "Has this kind of failure happened before?"},
            },
        },
        "gold": {"severity": "2", "cause": "power", "repeat": "true"},
    },
    {
        "id": "own_email_03", "kind": "email thread",
        "request": {
            "state": "From: Ines Halloway\nTo: Dev Ramacharan\nSubject: Re: venue walkthrough\n\nDev, thanks for the options. Thursday the 9th doesn't work for me because I'm presenting at the regional review that afternoon. Friday the 10th at 10:00 is fine, and so is Monday the 13th after lunch. Let's go with Friday unless the venue can't do mornings.\n\nInes\n\n> From: Dev Ramacharan\n> Could we do the walkthrough of Pellucid Hall on Thursday the 9th at 15:00, Friday the 10th at 10:00, or Monday the 13th at 14:00? The hall's coordinator needs an answer by Wednesday.",
            "questions": {
                "slot": {"type": "choice", "instructions": "Which slot does Ines choose?",
                         "criteria": {"thu": "Thursday the 9th at 15:00", "fri": "Friday the 10th at 10:00", "mon": "Monday the 13th at 14:00"}},
                "rules_out_thursday": {"type": "noul", "instructions": "Does Ines say she cannot make Thursday?",
                                       "criteria": {"true": "She rules Thursday out", "false": "She accepts Thursday or does not mention it"}},
                "next_step": {"type": "choice",
                              "criteria": {"confirm_friday": "Dev confirms the Friday 10:00 walkthrough with the hall",
                                           "propose_new": "Dev proposes new dates", "cancel": "Dev cancels the walkthrough",
                                           "wait": "Dev waits for Ines to choose a slot"}},
            },
        },
        "gold": {"slot": "fri", "rules_out_thursday": "true", "next_step": "confirm_friday"},
    },
    {
        "id": "own_meeting_04", "kind": "meeting notes",
        "request": {
            "state": "Weekly sync, Kestrelmoor Clinic front desk, 6 October.\nAttendees: Priya Okonkwo (lead), Sam Viljoen, Lotte Brandvold.\n1. Phone queue: the average wait rose to 7 minutes. Sam will draft a call-back script by Thursday.\n2. New check-in tablets: Lotte reported two tablets freezing at login. Decision: keep the paper fallback until the vendor's patch arrives.\n3. Holiday rota: postponed to next week because Priya is still collecting availability.\nNext meeting: 13 October.",
            "questions": {
                "script_owner": {"type": "choice", "instructions": "Who will draft the call-back script?",
                                 "criteria": {"priya": "Priya Okonkwo", "sam": "Sam Viljoen", "lotte": "Lotte Brandvold"}},
                "tablet_decision": {"type": "noul", "instructions": "Was a decision made about the check-in tablets?"},
                "rota": {"type": "choice", "instructions": "What happened to the holiday rota item?",
                         "criteria": {"agreed": "A rota was agreed", "postponed": "It was moved to a later meeting", "dropped": "It was removed from the agenda"}},
            },
        },
        "gold": {"script_owner": "sam", "tablet_decision": "true", "rota": "postponed"},
    },
    {
        "id": "own_review_05", "kind": "product review",
        "request": {
            "state": "Review of the Quillo K7 electric kettle, by a verified buyer.\n\nThe kettle looks great on the counter and the 1.7 litre size is right for our family. But the lid hinge snapped after three weeks of normal use, and now the lid won't stay closed, so the kettle doesn't switch itself off when the water boils. Customer service took nine days to reply. Heating speed is fine and it's quiet. I wouldn't buy it again.",
            "questions": {
                "stars": {"type": "score", "instructions": "What star rating best fits this review?",
                          "criteria": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]},
                "main_issue": {"type": "choice", "instructions": "What is the reviewer's main complaint?",
                               "criteria": {"price": "Too expensive", "durability": "A part broke or wore out", "noise": "Too loud",
                                            "speed": "Too slow", "size": "Wrong size", "design": "Looks or style",
                                            "delivery": "Shipping or packaging", "manual": "Instructions or setup",
                                            "support": "Customer service", "taste": "Taste or smell of the water"}},
                "would_rebuy": {"type": "noul", "instructions": "Would the reviewer buy this product again?",
                                "criteria": {"true": "They say they would buy it again or recommend it", "false": "They say they would not"}},
            },
        },
        "gold": {"stars": "1", "main_issue": "durability", "would_rebuy": "false"},
    },
    {
        "id": "own_order_06", "kind": "JSON order",
        "request": {
            "state": {"order_id": "VT-55120",
                      "customer": {"name": "Hollis Marrowby", "tier": "silver"},
                      "items": [{"sku": "LMP-02", "name": "Thistlebeam desk lamp", "qty": 1, "unit_price": 64.9},
                                {"sku": "BLB-10", "name": "Spare bulb pack", "qty": 2, "unit_price": 7.5}],
                      "subtotal": 79.9,
                      "shipping_address": {"country": "Canada"},
                      "requested_shipping": "standard",
                      "shipping_policy": "Free standard shipping on orders with a subtotal of 75.00 or more, for addresses in the United States only."},
            "questions": {
                "free_shipping": {"type": "noul", "instructions": "Does this order qualify for free shipping under the policy?",
                                  "criteria": {"true": "It meets every condition of the policy", "false": "It fails at least one condition"}},
                "method": {"type": "choice", "instructions": "Which shipping method did the customer request?",
                           "criteria": {"standard": "Standard", "express": "Express", "pickup": "Store pickup"}},
            },
        },
        "gold": {"free_shipping": "false", "method": "standard"},
    },
    {
        "id": "own_invoice_07", "kind": "JSON invoice",
        "request": {
            "state": {"invoice": "INV-2026-0412", "issuer": "Vellumtide Logistics", "client": "Brackenford Pottery",
                      "issued": "2026-08-01", "due": "2026-08-31", "today": "2026-09-20", "amount_due": 1840.0,
                      "payments": [{"date": "2026-08-28", "amount": 1000.0}],
                      "late_fee_policy": "A 2% late fee applies to any balance unpaid after the due date."},
            "questions": {
                "status": {"type": "choice", "instructions": "What is the payment status of this invoice?",
                           "criteria": {"paid": "Paid in full", "partial": "Partly paid", "unpaid": "Nothing paid", "disputed": "The client disputes it"}},
                "overdue": {"type": "noul", "instructions": "Is any part of the balance overdue today?"},
                "days_late": {"type": "score", "instructions": "How far past the due date is the invoice today?",
                              "criteria": ["Not past due", "1 to 14 days", "15 to 30 days", "More than 30 days"]},
            },
        },
        "gold": {"status": "partial", "overdue": "true", "days_late": "2"},
    },
    {
        "id": "own_sensor_08", "kind": "JSON sensor readings",
        "request": {
            "state": {"site": "Greenhouse 4, Saltmere Growers", "time": "2026-09-30T14:00",
                      "limits": {"temperature_c": [18, 30], "humidity_pct": [50, 85], "co2_ppm": [400, 1200], "soil_moisture_pct": [30, 60]},
                      "readings": {"temperature_c": 27.5, "humidity_pct": 91, "co2_ppm": 980, "soil_moisture_pct": 44}},
            "questions": {
                "out_of_range": {"type": "choice", "instructions": "Which reading is outside its limits?",
                                 "criteria": {"temperature": "Temperature", "humidity": "Humidity", "co2": "CO2", "soil": "Soil moisture"}},
                "alert": {"type": "score", "instructions": "Which alert level fits these readings?",
                          "criteria": ["No alert: all readings within limits", "Warning: one reading outside its limits",
                                       "Alarm: two or more readings outside their limits"]},
                "all_ok": {"type": "noul", "instructions": "Are all readings within their limits?",
                           "criteria": {"true": "Every reading is inside its range", "false": "At least one reading is outside its range"}},
            },
        },
        "gold": {"out_of_range": "humidity", "alert": "1", "all_ok": "false"},
    },
    {
        "id": "own_fiveq_09", "kind": "delivery note, five questions in one request",
        "request": {
            "state": "Delivery note from Ferngate Couriers, driver Pell Arno: parcel FG-7731 for Ada Merriweather was left at the side door of the house at 16:40 on Tuesday because nobody answered the bell or the phone. The box was dented on one corner but the seal was intact. The customer had asked for a signature on delivery when she booked. A photo of the parcel at the door is attached to this note for the customer's records.",
            "questions": {
                "signed": {"type": "noul", "instructions": "Was a signature collected from anyone at the address?",
                           "criteria": {"true": "Someone at the address signed for the parcel", "false": "Nobody signed for the parcel"}},
                "where": {"type": "choice", "instructions": "Where did the driver leave the parcel?",
                          "criteria": {"front": "At the front door", "side": "At the side door", "neighbour": "With a neighbour", "depot": "Back at the depot"}},
                "damaged": {"type": "noul", "instructions": "Was the outside of the box damaged in any way?",
                            "criteria": {"true": "The note mentions damage to the box", "false": "The box arrived undamaged"}},
                "request_followed": {"type": "score", "instructions": "How well did the delivery follow what the customer asked for?",
                                     "criteria": ["Fully followed", "Partly followed", "Not followed at all"]},
                "photo": {"type": "choice", "instructions": "Does the note come with a photo of the parcel?", "criteria": {"yes": "A photo is attached", "no": "There is no photo"}},
            },
        },
        "gold": {"signed": "false", "where": "side", "damaged": "true", "request_followed": "2", "photo": "yes"},
    },
    {
        "id": "own_long_log_10", "kind": "long state: service log",
        "request": {
            "state": LOG_STATE,
            "questions": {
                "first_failure": {"type": "choice", "instructions": "Which component failed first?",
                                  "criteria": {"edge_lb": "edge-lb", "auth_gateway": "auth-gateway", "billing_api": "billing-api",
                                               "queue_relay": "queue-relay", "db_replica_2": "db-replica-2", "db_primary": "db-primary"}},
                "impact": {"type": "score", "instructions": "How much were customers affected?",
                           "criteria": ["Not at all", "Slower responses only", "Some requests failed", "Complete outage"]},
                "resolved": {"type": "noul", "instructions": "Was the problem resolved by the end of the log?",
                             "criteria": {"true": "The last entries show normal operation", "false": "Errors continue at the end"}},
            },
        },
        "gold": {"first_failure": "db_replica_2", "impact": "2", "resolved": "true"},
    },
    {
        "id": "own_long_contract_11", "kind": "long state: contract excerpt",
        "request": {
            "state": CONTRACT_STATE,
            "questions": {
                "convenience_notice": {"type": "choice", "instructions": "How much notice must the Client give to end the agreement for convenience?",
                                       "criteria": {"d30": "30 days", "d60": "60 days", "d90": "90 days", "none": "It cannot be ended for convenience"}},
                "liability_cap": {"type": "noul", "instructions": "Does the agreement limit each party's total liability to a maximum amount?",
                                  "criteria": {"true": "A maximum is set", "false": "No maximum is set"}},
                "initial_term": {"type": "score", "instructions": "How long is the initial term?",
                                 "criteria": ["Less than 1 year", "1 year", "2 years", "More than 2 years"]},
            },
        },
        "gold": {"convenience_notice": "d90", "liability_cap": "true", "initial_term": "2"},
    },
    {
        "id": "own_long_minutes_12", "kind": "long state: board minutes",
        "request": {
            "state": MINUTES_STATE,
            "questions": {
                "irrigation": {"type": "choice", "instructions": "What did the board do with the irrigation budget proposal?",
                               "criteria": {"approved": "Approved it", "rejected": "Rejected it", "deferred": "Postponed the decision"}},
                "chair": {"type": "choice", "instructions": "Who chaired the meeting?",
                          "criteria": {"thorley": "Wenna Thorley", "pennock": "Idris Pennock", "agbaje": "Marisol Agbaje", "vessey": "Dorran Vessey"}},
                "lease_unanimous": {"type": "noul", "instructions": "Was the greenhouse 4 lease renewal approved unanimously?"},
            },
        },
        "gold": {"irrigation": "deferred", "chair": "thorley", "lease_unanimous": "false"},
    },
]
