# Privacy as Commodity

Deployable research showcase and multi-user privacy procurement demonstrator.
Source manuscript: the authors' `main_final2/p2v2_version/` archive (2026-10-25 release).
The public PDF snapshot is included in `site/papers/`; the reproducibility
package is in [`../reproducibility/`](../reproducibility/README.md).

Live: https://privacy-as-commodity.pages.dev/ ·
[Comparisons](https://privacy-as-commodity.pages.dev/results) ·
[Shared auction](https://privacy-as-commodity.pages.dev/auction).
See `GUIDE_zh.md` for a Chinese demonstration guide and `DEPLOYMENT.md` for
release URLs and validation evidence.

## Run and deploy

Requires Node.js 22+ and a Cloudflare account with Pages and D1 access.

```sh
npm ci
npm run db:local
npm run dev
```

Open `http://127.0.0.1:8091`. The same app runs as Cloudflare Pages assets plus
a Pages Function, with durable shared room state in D1. No local server is
needed once deployed.

For a fresh Cloudflare account:

```sh
npx wrangler login --device --scopes account:read user:read pages:write d1:write
npx wrangler d1 create privacy-commodity-auctions
```

Put your returned database ID in `wrangler.toml` (the public source contains a
placeholder). Choose your own project name and update the `deploy` script in
`package.json` if it differs. Create your Pages project, then initialize and
deploy:

```sh
npx wrangler pages project create privacy-as-commodity --production-branch main
npm run db:remote
npm run deploy
```

Do not publish the repository root. `site/` is the explicit publication
directory. Wrangler bundles `_worker.js` with the server code and frozen
weights in `lib/`. API credentials, test fixtures and private room access keys
are not static assets. Keep Wrangler credentials outside this project.

## Participant workflow

1. A host creates a room with a supported budget and task parameter κ.
2. Participants open the invite URL on separate browsers/devices and register
   an institution name and synthetic record count (10–500).
3. The host verifies every record count and opens bidding. Registration freezes.
4. Each participant privately submits a unit valuation [0,1] and capacity
   [0.1,5]. Offers may be updated until clearing.
5. The host clears when everyone has bid. Server inference uses the actual
   frozen MFG-RegretNet checkpoint, the frozen τ table, exact capped
   projection, 65 valuation grid points, and the data dividend.
6. Each participant sees compensation, dividend and utility at the reported
   cost. Acceptance confirms a synthetic contribution and records demo
   credits. Rejection settles to zero with no reallocation. Receipts can be
   downloaded. Repeated acceptance cannot credit twice.

The host may also register as a participant. Rooms support 2–10 institutions;
the paper's accuracy evaluation used N=10. Rooms expire after seven days and
expired records are purged on the next room creation.

## Security and persistence

- Independent, unguessable capability tokens for the host and each participant.
  Only SHA-256 token hashes are stored in the server room state.
- Tokens stay in that browser's local storage and can be explicitly downloaded
  as a **private** access file and restored from the auction landing page on
  another device. Invite URLs contain only a room ID. Treat a
  private access key as a credential; an invite link is not a host key.
- Server-side authorization checks every write; participant reports and private
  cost/utility are not returned to other participants, the host or observers.
  Public allocations and payments follow clearing. Raw embedding values are
  not exposed.
- D1 compare-and-swap updates serialize concurrent joins, bids, clearing and
  settlement; duplicate clear/accept requests are idempotent.
- Same-origin JSON mutations, input bounds, request size checks, rate limits,
  text escaping and a restrictive content security policy.
- Monetary rewards are demo credits. There is no wallet custody, cash deposit,
  withdrawal or payment-provider integration. No private records/model updates
  are uploaded. The web service computes in plaintext on Cloudflare over HTTPS;
  it is **not** an implementation of the separately measured SMPC benchmark.

## Evidence and model provenance

The historical `scripts/export_evidence.py` reads the final paper's completed experiment
artifacts. It exports all 18 settings, the proposed arm and 10 external arms,
sample SDs, three seed values, per-client cumulative privacy, incentive results
and secure costs. Missing/unreachable comparisons stay missing. No training or
experiment is rerun, and no manuscript file is modified. It requires the full
research archive and is retained for provenance; the public clone already
contains the generated assets and model. It does not need that archive to run
or deploy. Run `npm run test:evidence` from this directory to verify the public
release's hashes and displayed comparison arithmetic.

The export script uses the immutable, hash-checked SMPC reference to load the
checkpoint. Public coefficients for six supported budgets and κ in {3,30}
are exported together with the positive encoder weights. The JS inference
implementation matches 36 Python reference profiles to <1e-8 absolute error.
Frozen checkpoint SHA-256:
`baddeed7201f2d4dfe82bce9aeb65a0fed86d29759618cab4a7130f57a2b382b`.

The public manifest `/data/provenance.json` records exact source hashes. Main
curves are manuscript images with its disclosed display smoothing. The all-
settings smoothed supplement images, unsmoothed images and raw per-round CSV
are also available. CSV endpoint values are not
smoothed. Manuscript and supplement PDFs are byte-identical source copies.

## Validation

```sh
npm test
# With the local Pages server running:
node tests/api-flow.mjs
node tests/full-room.mjs
# With agent-browser and Chromium installed:
node tests/browser-smoke.mjs
```

The HTTP test uses the actual local D1 binding: concurrent joins, state
transitions, forbidden actions, private bids, concurrent clearing, duplicate
acceptance, rejected settlement and persistent reload. It creates a clearly
named QA room. `TEST_URL` may point to an authorized staging deployment.

Browser checks generate screenshots in `qa/`, outside published assets. The
authors' private room-access files and prior QA directory are not distributed.
The site was tested at desktop and mobile widths using agent-browser.

## Connecting a real institution

The deployed app is a working collaboration and settlement prototype. To run
the complete real FL system, integrate an identity/organization provider,
independently verified registration counts, a participant-side DP/FL agent,
release verification, privacy ledger accounting, and payment-provider webhooks.
Settlement must then depend on verified release evidence instead of the demo
confirmation. Real-time repeated auctions need a persistent per-client privacy
ledger across rooms. Neither accuracy nor privacy guarantees can be inferred
from a participant clicking the demonstration confirmation.

Keep these connectors separate from the frozen mechanism so institutional
validation, model selection and money transfer can be reviewed independently.
Cloudflare quotas apply; the project does not enable a paid plan automatically.
