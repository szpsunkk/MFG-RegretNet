# Deployment record · 2026-10-25

Target: https://privacy-as-commodity.pages.dev/ (existing project).
Status: published using Cloudflare Pages and D1, without a paid-plan upgrade.

- Production: https://privacy-as-commodity.pages.dev/
- Final immutable release: https://4fcdb521.privacy-as-commodity.pages.dev/
- Tested preview: https://20592b98.privacy-as-commodity.pages.dev/
- D1 database: `privacy-commodity-auctions`
- The public source uses a placeholder D1 ID; provision your own database.
- Worker binding: `DB`; existing Pages project and production branch `main`.

## Delivered application

- 16 English pages, responsive dark interface, original branding and URL routes.
- All 18 confirmation settings, 11 mechanisms, three seed values and sample
  SDs; accuracy, composed privacy, incentives and five secure execution sizes.
- Final manuscript and supplement from `main_final2/p2v2_version/`, source
  figures, CSV download and file-hash manifest.
- Persistent 2–10-person auction rooms, invitations, private reports, host
  registration verification, model-based allocation and payments, acceptance
  or rejection, demo-credit settlement and downloadable receipts.
- Downloadable private capability files and access restoration across devices.

## Validation completed locally

- 36 Python/JavaScript reference profiles, maximum absolute difference
  9.325873406851315e-15. Budget, capacity and reported-cost IR invariants pass.
- Actual HTTP/D1 tests: concurrent registration, role restrictions, private
  bids, invalid inputs, frozen registration, idempotent concurrent clearing,
  duplicate acceptance, rejected zero settlement, reload and origin checks.
- Two isolated browser sessions completed registration → bidding → clearing →
  acceptance → recorded rewards. A third session restored host access using
  the private download.
- A full 10-participant HTTP/D1 room settled successfully. The local clearing
  request took about 15 ms (7 ms measured model inference); these are local
  checks, not claims about production latency.
- All 16 pages rendered at 1440 px and 390 px without horizontal page overflow.
- Room keyboard accessibility check: zero detected WCAG A/AA violations after
  making scrollable regions focusable. Automated contrast review is incomplete
  for gradients; desktop and mobile screenshots were visually checked.

QA screenshots and local access files remain in the authors' private archive and are excluded from the
published assets and distributable project archive. No test credentials belong
in a public artifact.

## Evidence boundaries

CIFAR-10 leads in 8/8 confirmation settings, average +1.70075 percentage points;
MNIST leads in 5/10, average +0.0371333 points. The website retains losses and
unreachable targets and explains the aggregation control. Accuracy comes from
MNIST/CIFAR-10, not the intrusion-detection application scenario.

Rewards are demo credits and acceptance is a synthetic contribution. The app
does not transfer real money, run real federated training, or deploy the
separately measured SMPC protocol. Server-side HTTPS computation sees reports;
other room users and the host cannot retrieve others' raw reports through the
API. A mean embedding is not presented as a computed Nash equilibrium.

## Publication and rollback

Published only `site/`, bundling `lib/` into the server worker. Initialized the
new dedicated D1 database using `schema.sql`. Preview deployment passed:

- HTTP concurrency/security/settlement integration test.
- Full 10-institution room, all 10 acceptances, 29.356170268322092 credits
  within B=30. Observed clearing HTTP round trip was about 407 ms through this
  machine's network; this is not a performance guarantee or a CPU benchmark.
- All 16 browser routes at desktop and mobile widths, 32 checks in total.
- Byte-level verification of 31 published assets, including both PDFs, result
  JSON, source manifest, raw CSV and figures; API health and CSP also checked.

The production hostname separately passed the HTTP integration test and asset
hash verification. Its public room view displayed the 10-participant settled
market without script errors. The final PDF-only refresh changed three assets
(two PDFs and provenance); application code and experimental data were unchanged.

PDF source files were still being recompiled by other work during publication.
The final release captures the latest complete files read at 11:59 JST;
each manifest hash is computed from the exact bytes copied, and the source
remained consistent during the capture. Published SHA-256 values:

- Main: `9d30eece8b3e108e43154b2a53e014430ee7302bf8ac443680d951580a2d7dfb`
- Supplement: `d9165f90a04fd4d39a900a57d99eb4011b397763afacdd8a39e836f2be3fa308`

QA rooms have synthetic names/data and expire normally after seven days.
No private participant tokens are included in this record or the source archive.
One PDF upload request stalled during the last refresh and was terminated and
retried using the same captured files. The already published site stayed live.
Local development and browser test sessions were closed after cloud checks.

Previous immutable deployment retained for rollback:
https://581ce595.privacy-as-commodity.pages.dev/

Previous publication source remains in the authors' archive. No paper or
experiment source was edited by this deployment. Source PDFs may be updated
independently, so a later release requires a new export and hash check.
