"""
Sample Insurance Documents
===========================
Realistic P&C and Reinsurance text for demo/testing.
Covers: Policy wording, Reinsurance treaty slip, Exclusions schedule.
"""

SAMPLE_DOCS = {
    "property_policy": """
# COMMERCIAL PROPERTY INSURANCE POLICY
Policy Number: CPP-2024-MH-00892
Insured: Hamilton Logistics Pvt Ltd
Line of Business: Property & Casualty
Effective Date: 01 January 2024
Expiry Date: 31 December 2024

## SECTION 1: DECLARATIONS

The insurer agrees to provide coverage as described herein subject to the terms, conditions, and exclusions set forth in this policy.

| Coverage | Sum Insured | Premium Rate | Annual Premium |
|---|---|---|---|
| Building | $10,000,000 | 0.15% | $15,000 |
| Contents | $5,000,000 | 0.20% | $10,000 |
| Business Interruption | $2,000,000 | 0.25% | $5,000 |
| Total | $17,000,000 | — | $30,000 |

## SECTION 2: DEFINITIONS

2.1 Insured Property means the building, fixtures, fittings, plant, machinery, and contents described in the Schedule, situated at the premises specified therein.

2.2 Occurrence means any one event or series of events arising from a single originating cause.

2.3 Deductible means the amount to be borne by the Insured in respect of each and every loss occurrence, as specified in Section 5 of this Policy. The deductible applicable to flood and earthquake peril is $500,000.

2.4 Sum Insured means the maximum liability of the Insurer in respect of any one occurrence, as stated in the Declarations.

## SECTION 3: COVERAGE

3.1 All Risks Coverage
This Policy covers all risks of direct physical loss or damage to the Insured Property from any cause whatsoever, except as specifically excluded under Section 4.

3.2 Named Perils — Extended Coverage
The following named perils are expressly covered under this Policy:
- Fire, lightning, and explosion
- Storm, tempest, windstorm, and cyclone
- Flood and inundation (subject to $500,000 deductible)
- Earthquake and volcanic eruption (subject to $500,000 deductible)
- Impact by aircraft, vehicles, or vessels
- Malicious damage and vandalism

3.3 Business Interruption
In the event of an insured loss causing an interruption to the business, the Insurer shall indemnify the Insured for:
(a) Net profit that would have been earned
(b) Standing charges and fixed expenses continuing during the indemnity period
Maximum indemnity period: 12 months
Maximum liability: $2,000,000

## SECTION 4: EXCLUSIONS

4.1 This Policy does not cover loss, damage, or liability directly or indirectly caused by, contributed to by, resulting from, or in connection with:

(a) War, invasion, acts of foreign enemies, hostilities, civil war, rebellion, terrorism
(b) Nuclear, radiological, chemical, or biological contamination
(c) Cyber attack, hacking, or malicious code causing physical damage (see Cyber Exclusion Clause CL380)
(d) Gradual deterioration, wear and tear, inherent vice, or latent defect
(e) Loss of market, delay, loss of use, or consequential loss except as provided under Section 3.3
(f) Wilful act or gross negligence of the Insured

4.2 Cyber Exclusion — CL380
Notwithstanding any provision to the contrary, this Policy excludes any loss, damage, liability, cost, or expense of whatsoever nature directly or indirectly caused by, contributed to by, resulting from, or arising out of or in connection with any cyber attack or act of cyber terrorism. This exclusion applies regardless of any other cause or event contributing concurrently or in any other sequence to the loss.

## SECTION 5: CONDITIONS

5.1 Deductible Schedule
| Peril | Deductible |
|---|---|
| All perils (general) | $50,000 |
| Flood / Inundation | $500,000 |
| Earthquake | $500,000 |
| Windstorm / Cyclone | $100,000 |

5.2 Basis of Settlement
In the event of loss or damage, the basis of settlement shall be:
(a) Buildings: Reinstatement value (cost of rebuilding to the same specification)
(b) Contents: Replacement value (cost of replacement with new item of like kind)
(c) Machinery: Replacement value less depreciation not exceeding 50%

5.3 Claims Notification
The Insured shall notify the Insurer within 7 days of discovery of any loss or damage likely to give rise to a claim. Failure to notify within 30 days shall render this policy void with respect to such claim.

5.4 Co-Insurance Clause
If the Sum Insured is less than the actual value of the Insured Property at the time of loss, the Insured shall be considered as being their own Insurer for the difference and shall bear a rateable proportion of any loss accordingly.
""",

    "reinsurance_treaty": """
# REINSURANCE TREATY — PLACEMENT SLIP
Treaty Reference: XL-PROP-2024-HAM-001
Reinsured: Hamilton Re Ltd
Line of Business: Property Excess of Loss
Class: Non-Marine Property XL

## TREATY STRUCTURE

| Layer | Limit | Attachment Point | Rate on Line |
|---|---|---|---|
| Layer 1 (Primary XL) | $5,000,000 xs $5,000,000 | $5,000,000 | 12.5% |
| Layer 2 (Upper XL) | $10,000,000 xs $10,000,000 | $10,000,000 | 7.8% |
| Layer 3 (Catastrophe) | $25,000,000 xs $20,000,000 | $20,000,000 | 4.2% |

## SECTION A: BUSINESS COVERED

A.1 This Treaty covers the Reinsured's net retained liability in respect of all policies in force at inception and all policies issued or renewed during the period of this Treaty, covering:
- Commercial and industrial property (all risks and named perils)
- Business interruption and consequential loss as a consequence of property damage
- Engineering and construction (on-risk policies only)

A.2 Territorial scope: India, Bangladesh, Sri Lanka, and Nepal (IBSN Zone)

## SECTION B: ATTACHMENT POINT AND LIMIT

B.1 The Reinsurer agrees to indemnify the Reinsured for each and every loss occurrence in excess of the Attachment Point of $5,000,000 up to the Reinsurer's Limit of $5,000,000 per occurrence, making Layer 1 capacity of $5,000,000 xs $5,000,000.

B.2 The maximum liability of the Reinsurer shall not exceed $5,000,000 in respect of any one occurrence, nor $10,000,000 in the aggregate during the Treaty Period.

B.3 Annual Aggregate Deductible: $2,500,000 (applies to Layer 1 only)

## SECTION C: EXCLUSIONS

C.1 This Treaty excludes the following:
(a) Nuclear incident, contamination, or radioactive fallout
(b) War, terrorism, and political risks
(c) Flood losses in excess of $15,000,000 per occurrence (catastrophic flood sublimit applies)
(d) Cyber-induced physical damage losses (per LSW 3001 wording)
(e) Losses arising from named windstorms where the Named Storm Event Deductible of $1,000,000 applies

C.2 Cyber Exclusion — LSW 3001
This Treaty is subject to the Cyber and Information Technology Exclusion Clause (LSW 3001). Losses directly or indirectly arising out of, or contributed to by, or resulting from Cyber Operations are excluded unless physical damage resulted independently of any Cyber Operation.

## SECTION D: PREMIUM

D.1 Minimum and Deposit Premium: $1,875,000 (payable in four quarterly instalments of $468,750)
D.2 Adjustable Premium: Based on Reinsured's subject premium income. Rate: 1.875% on subject net earned premium income.
D.3 Minimum Premium: $1,500,000 (75% of estimated deposit)

## SECTION E: LOSS SETTLEMENT

E.1 Claims Cooperation Clause: The Reinsured shall advise the Reinsurer of all losses likely to exceed $2,500,000 within 30 days of notification.

E.2 Cash Loss Clause: The Reinsurer shall pay any claim within 30 days of receipt of a signed proof of loss, provided the loss exceeds $1,000,000.

E.3 Follow the Fortunes: The Reinsurer agrees to follow the fortunes of the Reinsured in all matters pertaining to this Treaty, provided the Reinsured acts in good faith and in accordance with sound underwriting principles.
""",

    "claims_procedure": """
# CLAIMS PROCEDURE MANUAL — PROPERTY & CASUALTY
Document Type: Internal SOP
Line of Business: Property & Casualty / Reinsurance
Version: 4.2 | Effective: March 2024

## SECTION 1: FIRST NOTICE OF LOSS (FNOL)

1.1 Upon receipt of an FNOL, the Claims Handler shall:
(a) Assign a unique claim reference number (format: CLM-YYYY-LOB-NNNNNN)
(b) Acknowledge receipt to the Insured within 24 hours
(c) Classify the loss by peril, estimated quantum, and coverage section
(d) For estimated losses exceeding $500,000, appoint an Independent Loss Adjuster within 48 hours

1.2 FNOL Triage Matrix
| Estimated Loss | SLA — First Response | SLA — Adjuster | Authority Level |
|---|---|---|---|
| Under $50,000 | 24 hours | Not required | Junior Handler |
| $50,000 – $500,000 | 24 hours | 5 working days | Senior Handler |
| $500,000 – $2,000,000 | 4 hours | 48 hours | Claims Manager |
| Above $2,000,000 | 1 hour | 24 hours | Chief Claims Officer |

## SECTION 2: INVESTIGATION AND ADJUSTMENT

2.1 The appointed Loss Adjuster shall submit a Preliminary Report within 10 working days of appointment, covering:
- Cause and circumstances of loss
- Policy coverage and applicable exclusions
- Preliminary loss estimate
- Reserve recommendation

2.2 For Business Interruption losses, the Loss Adjuster shall appoint a Forensic Accountant to review:
- Historical financial statements (3 years minimum)
- Order books and sales records
- Fixed overhead schedule
- Net profit calculation per policy definition

2.3 Subrogation: Where the Insurer has paid a claim, the right of subrogation vests in the Insurer. The Claims Handler must flag all losses where a liable third party may be identified, within 30 days of loss settlement.

## SECTION 3: RESERVING GUIDELINES

3.1 Initial Reserve Philosophy: Reserves shall be set on a realistic best-estimate basis, not a worst-case basis.

3.2 Reserve Adequacy Review: All reserves exceeding $1,000,000 shall be reviewed monthly. Reserves exceeding $5,000,000 shall be reviewed weekly.

3.3 Incurred But Not Reported (IBNR) Reserve: The actuarial IBNR reserve shall be calculated quarterly using the Chain-Ladder Method and Bornhuetter-Ferguson Method.

## SECTION 4: REINSURANCE RECOVERY

4.1 Where a loss is expected to breach the Reinsurance Attachment Point, the Claims Handler shall:
(a) Notify the Reinsurance team within 24 hours of identifying the breach potential
(b) Issue a loss advice to all relevant reinsurers
(c) Seek reinsurer concurrence for settlements exceeding $500,000

4.2 Reinsurance Bordereaux: Monthly bordereaux shall be submitted to reinsurers for all outstanding losses and recoveries.
"""
}


def get_sample_queries():
    """Predefined test queries with expected sections (for evaluation)."""
    return [
        {
            "query": "What is the deductible for flood damage?",
            "relevant_sections": ["SECTION 5: CONDITIONS", "SECTION 2: DEFINITIONS"],
        },
        {
            "query": "Does this policy cover cyber-induced business interruption losses?",
            "relevant_sections": ["SECTION 4: EXCLUSIONS", "SECTION 3: COVERAGE"],
        },
        {
            "query": "What is the attachment point for the reinsurance Layer 1?",
            "relevant_sections": ["SECTION B: ATTACHMENT POINT AND LIMIT", "TREATY STRUCTURE"],
        },
        {
            "query": "How soon must claims be notified to the insurer?",
            "relevant_sections": ["SECTION 5: CONDITIONS", "SECTION 1: FIRST NOTICE OF LOSS (FNOL)"],
        },
        {
            "query": "What is the maximum liability under the reinsurance treaty?",
            "relevant_sections": ["SECTION B: ATTACHMENT POINT AND LIMIT"],
        },
        {
            "query": "What are the annual aggregate limits in the reinsurance treaty?",
            "relevant_sections": ["SECTION B: ATTACHMENT POINT AND LIMIT"],
        },
        {
            "query": "What is the basis of settlement for building losses?",
            "relevant_sections": ["SECTION 5: CONDITIONS"],
        },
        {
            "query": "Who has authority to handle claims above $2 million?",
            "relevant_sections": ["SECTION 1: FIRST NOTICE OF LOSS (FNOL)"],
        },
    ]
