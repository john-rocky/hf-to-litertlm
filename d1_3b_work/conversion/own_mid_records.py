"""One mid-length record written for the d1-3B port in round 2: `own_mid_08k_001`, a plain-text state of about 800
tokens, so that its rows fall in the 513-1,024-token bucket (the round-1 fixture had no row there). It names no person,
organisation, product or place: people are roles, machines are numbered. One noul, one choice and one score question;
`gold` is the answer the log was written to have (keys as in the Kev fixtures). Token counts are in fixtures/rows.json.
"""

LOG_STATE = """Overnight production log, bread line, shift from 22:00 to 06:00. Entries are written by the night supervisor unless another role is named.

22:00 Shift handover. Three ovens on the line, all at set point. Oven 1 at 220 C, oven 2 at 230 C, oven 3 at 210 C. Flour silo at 64 percent.
22:10 Mixer 1 started dough for batch 1 (white sandwich loaves, 180 loaves). Mixer 2 started dough for batch 2 (wholemeal, 150 loaves).
22:50 Batch 3 (rye, 120 loaves) mixed on mixer 1.
23:05 Batch 1 loaded into oven 1. Batch 2 into the proofer.
23:40 Batch 2 loaded into oven 2. Batch 3 into the proofer.
23:55 The mixer operator reported a grinding noise from mixer 2. Mixer 2 stopped for a check; the maintenance technician found a loose guard bolt and tightened it. Mixer 2 back in use at 00:10.
00:15 Batch 2 out of oven 2, core temperature 96 C. Batch 3 loaded into oven 3.
00:30 Oven 2 showed a high-temperature warning (242 C against a set point of 230 C). The warning cleared by itself after four minutes; the oven stayed in use. Logged for the day shift.
01:00 Batches 4 to 8 mixed and proofed without remarks; batches 4 and 5 baked in oven 1, batches 6 and 7 in oven 2.
01:10 Oven 3 burner fault. The flame went out and the oven would not relight. Batch 8, which had been loaded at 01:02, was taken out early.
01:20 The maintenance technician started work on oven 3. Batch 8 moved to oven 1 to finish baking.
02:00 Batch 9 (white sandwich loaves) out of oven 2 with a dark crust on one tray. Batch 9 held for a second check by the quality technician.
02:20 The quality technician cut six loaves from batch 9: the crumb was fine, the dark tray was trimmed. Batch 9 released.
02:30 Oven 3 relit after a new ignition electrode was fitted. Set point 210 C reached at 02:55.
03:00 Batches 10 to 20 mixed, proofed and baked across the three ovens without remarks.
04:05 Batch 21 (wholemeal) loaded into oven 3. At 04:12 oven 3 dropped to 150 C; the burner had gone out again.
04:15 Batch 21 out of oven 3 under-baked, core temperature 71 C. Batch 21 thrown away, 150 loaves.
04:25 Oven 3 taken out of use for the rest of the shift. Remaining batches split between ovens 1 and 2.
05:10 Batches 22 to 26 baked in ovens 1 and 2 without remarks.
05:30 Batch 27 (rye) out of oven 1 with a split crust on most loaves. Batch 27 held for a second check by the quality technician.
05:45 The quality technician weighed and cut ten loaves from batch 27: weight and crumb within limits, the split is cosmetic. Batch 27 released.
06:00 Shift end. Batches made: 27. Batches released: 26. Oven 3 out of use; the day shift to call the burner service. Mixer 2 guard bolt to be checked again."""

OWN_MID = [
    {
        "id": "own_mid_08k_001", "kind": "mid-length state (about 800 tokens): overnight production log",
        "request": {
            "state": LOG_STATE,
            "questions": {
                "discarded": {"type": "noul", "instructions": "Was any batch thrown away during the shift?"},
                "first_fault": {"type": "choice", "instructions": "Which oven broke down first during the shift?",
                                "criteria": {"oven_1": "Oven 1", "oven_2": "Oven 2", "oven_3": "Oven 3",
                                             "none": "No oven broke down"}},
                "second_checks": {"type": "score", "instructions": "How many batches were held for a second check?",
                                  "criteria": ["None", "One", "Two", "Three or more"]},
            },
        },
        "gold": {"discarded": "true", "first_fault": "oven_3", "second_checks": "2"},
    },
]
