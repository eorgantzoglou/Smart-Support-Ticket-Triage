# Label Taxonomy — Airline Support Ticket Triage

Every incoming customer message receives exactly **one intent label**, one **urgency level**, and one **abusive flag**. This taxonomy was derived from manual review of real tweets in the dataset (see `notebooks/01_exploration.ipynb`) and is used verbatim in the LLM labeling prompt.

## Labeling rules

1. **Single label — dominant intent.** When a message contains multiple issues, choose the one that most needs action *now*.
   *Worked example:* "we're trapped in Dublin til tomorrow after missed connection... Bags confirmed arrived, can't be located" mentions both a disruption and baggage; it is labeled `delay_disruption` because the stranded passenger is the dominant, time-critical issue.
2. **Judge the text alone.** The classifier will see one message with no conversation history. The question is never "is this a reply?" but "does this text, by itself, carry a classifiable intent?" If not → `other_unclear`.
3. **Customer vs non-customer.** `spam_irrelevant` means the author is not a customer seeking support (news bots, jokes, trolling). `other_unclear` means a real customer whose message cannot be classified on its own. These route differently: spam is discarded, unclear goes to a human.
4. **Specific beats general.** `general_question` and `general_complaint` are fallbacks: use them only when no specific class fits. *"When can I buy the cat ticket?"* is a question, but it belongs to `special_assistance` because a specific class applies.

## Intent classes (10)

### `delay_disruption`
Flight delays, missed connections, or in-airport waiting caused by the airline's operations.
- *"Waiting on @AmericanAir flight attendants to get to Phoenix. Offered to push the cart, no dice. Guess we'll wait."*
- *"And, of course, now that we're late despite our 20min early wheels down, our connections are at risk."*
- *"Yes, we're trapped in Dublin til tomorrow after missed connection due to late BA arrival yesterday."*

### `checkin_boarding_issue`
Problems during check-in or boarding: denied boarding, gate confusion, being wrongly marked late.
- *"And CATHY is telling me I'm late when A. I checked in Emary B. I came an hour before my flight."*

### `flight_cancellation_rebooking`
Cancelled flights and the rebooking or compensation that follows (vouchers, alternative routes).
- *"y'all cancelled my flight and rebooked at a DIFFERENT AIRPORT w no travel or food vouchers"*

### `lost_luggage`
Lost, delayed, or damaged baggage and attempts to track or recover it.
- *"my bags are lost and I have no way to start getting them rerouted without leaving security"*

### `special_assistance`
Requests about special needs: pets, dietary requirements, accessibility, traveling with children.
- *"I need to travel with my cat. I got my ticket but not the cat one. When can I buy the cat ticket?"*
- *"I did, but was told that gluten free AND veggie is not an option."*

### `general_complaint`
Anger or criticism about the airline **without a specific actionable request** that fits a class above. Use only when nothing more specific applies — most messages contain complaint *tone*; that alone does not make them `general_complaint`.
- *"Apparently doesn't give two shits about people affected by the wild fires in Sonoma County. Never flying with them again."*
- *"why have I been standing at the baggage check Terminal 1 for 10 minutes with NO ATTENDANT IN SIGHT??"*
- *"Please fix your marketing. The e-mail you sent today says: no seat until you get to Gate..."*

### `general_question`
An information request (pricing, policies, procedures) with no underlying problem to solve, when no specific class fits.
- *"Sure, how much would an upgrade cost, out of curiosity?"*
- Borderline vs `general_complaint`: when a message both vents and asks, the dominant-intent rule decides — *"Please fix your marketing. The e-mail says: no seat until you get to Gate..."* stays `general_complaint` because the venting ("fix your marketing") is the main message, the question is secondary.

### `praise_feedback`
Positive feedback, thanks, or resolved-issue confirmations. Needs no routing to a support team.
- *"Thanks for the reply. We've got it sorted now."*
- *"Great article! No airline can be perfect, but the @AmericanAir social media team is a fantastic resource."*

### `spam_irrelevant`
Not a customer seeking support: news/marketing accounts, jokes, trolling, free-stuff begging.
- *"Issues National Travel Advisory... #TravelTuesday #TuesNews"* (news bot)
- *"hello can i get a free flight to london rn"*
- *"i want someone to love me as much as you love delta"*

### `other_unclear`
A real customer, but the message alone carries no classifiable intent (context-less replies, ambiguous fragments).
- *"It was yesterday, flight #1791 from Charlotte to EWR."*
- *"Just DMd you"*

## Urgency (3 levels)

The criterion is **time**: is travel currently in progress or imminent?

- **`high`** — the customer is traveling now, stranded, or flying within hours. *"trapped in Dublin til tomorrow"*, *"my connections are at risk"*.
- **`medium`** — an active issue needing resolution, but no immediate journey at stake. *"When can I buy the cat ticket?"*, *"my bags are lost"* (post-trip).
- **`low`** — venting, questions, feedback with no time pressure. *"Guess we'll wait."*, praise, spam.

## Abusive (boolean)

`true` only for **personal attacks**: insults or threats directed at a person (staff, agents) or slurs of any kind. Profanity aimed at the *situation* or the *company* is anger, not abuse.

- *"THIS IS BULLSHIT"* → `false` (profanity at the situation)
- *"doesn't give two shits about people"* → `false` (anger at the company)
- Insulting a named employee, threats, slurs → `true`

Abusive messages are escalated to a human regardless of intent.

## Provenance

Labels are produced by a local Qwen LLM (~35B, via Ollama) using this document as the prompt's class definitions. A random sample of 250–300 tweets is human-verified to form the gold test set; labeler–human agreement is reported in the README.
