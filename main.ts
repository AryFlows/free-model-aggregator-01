// The entry server: two client wires in, ten provider wires out.
//
// Startup reads the brief, probes every provider's list route, and builds the catalog
// that is both registered with the trial env and printed by --list. Then it serves:
//   GET  /v1/models              the free models, in the wire the caller speaks
//   POST /v1/chat/completions    OpenAI's shape, JSON or SSE
//   POST /v1/messages            Anthropic's shape, JSON or SSE
//
// Routing picks the highest-rated free model that meets the request's requirements
// (from the workload manifest, in arrival order, with a body-derived fallback), and
// fails over to the next candidate when a provider refuses — retrying the same
// provider once first, because a scheduled refusal is transient and the best model is
// worth one more call. A provider that has refused three times sorts last but is never
// skipped outright: a dead provider's refusal is cheap, and its budget may still hold
// a call. The server polls the env for work and exits 0 when the trial is done.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isFreeModel,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  ratingRowFor,
  readBrief,
  type Brief,
  type CatalogDoc,
  type Json,
  type Probe,
  type ProviderEntry,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicJsonAnswer,
  anthropicModelList,
  anthropicStreamToOpenAI,
  canonicalFromAnthropic,
  canonicalFromOpenAI,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  openAIError,
  openAIJsonAnswer,
  openAIModelList,
  openAIStreamToAnthropic,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  toAnthropic,
  toGemini,
  toOpenAI,
  type CRequest,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** One upstream call may take at most this long. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** How often the env is asked whether the trial is done. */
const POLL_MS = 100;
/** Largest client body accepted. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A provider sorts last once it has refused this many calls in the trial. */
const DEAD_AFTER_REFUSALS = 3;
/** Attempts per candidate before moving to the next one. */
const ATTEMPTS_PER_CANDIDATE = 2;

type ClientWire = 'openai' | 'anthropic';

interface ProviderState {
  calls: number;
  successes: number;
  refusals: number;
}

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
}

interface UpstreamOk {
  ok: true;
  answer: ProviderAnswer | null;
  stream: ReadableStream<Uint8Array> | null;
}

interface UpstreamFail {
  ok: false;
  status: number;
  error: string;
}

type UpstreamResult = UpstreamOk | UpstreamFail;

class Aggregator {
  readonly brief: Brief;
  readonly envUrl: string;
  readonly probes: Probe[];
  readonly catalog: CatalogDoc;
  readonly states = new Map<string, ProviderState>();
  requestCount = 0;
  inFlight = 0;

  constructor(brief: Brief, envUrl: string, probes: Probe[], catalog: CatalogDoc) {
    this.brief = brief;
    this.envUrl = envUrl;
    this.probes = probes;
    this.catalog = catalog;
  }

  stateOf(providerId: string): ProviderState {
    let state = this.states.get(providerId);
    if (state === undefined) {
      state = { calls: 0, successes: 0, refusals: 0 };
      this.states.set(providerId, state);
    }
    return state;
  }

  /** The free models, deduplicated by id and sorted, for GET /v1/models. */
  servedModels(): ServedModel[] {
    const byId = new Map<string, ServedModel>();
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingRowFor(this.brief.ratings, probe.provider.id, model.id);
        if (!isFreeModel(model, row)) continue;
        if (!byId.has(model.id)) {
          byId.set(model.id, { id: model.id, provider: probe.provider.id, displayName: model.displayName });
        }
      }
    }
    return [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }

  /** The requirements for the request at arrival index `index`: the manifest's row, or what the body itself shows. */
  requirementsFor(index: number, req: CRequest): Requirement {
    const entry = this.brief.workload[index];
    if (entry !== undefined) return entry.requires;
    const out: Requirement = {};
    if (req.tools.length > 0) out.tools = true;
    if (req.messages.some((m) => m.blocks.some((b) => b.kind === 'image'))) out.vision = true;
    if (req.jsonMode) out.json = true;
    return out;
  }

  /** Every free model that meets the requirements, best first. */
  candidatesFor(req: CRequest, requires: Requirement): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingRowFor(this.brief.ratings, probe.provider.id, model.id);
        if (!isFreeModel(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
        // An OpenAI JSON-mode request must not reach an Anthropic-wire provider:
        // that wire cannot carry JSON mode, so the echo's JSON check would fail.
        if (req.jsonMode && probe.provider.wire === 'anthropic') continue;
        out.push({ provider: probe.provider, model: model.id, rating: row?.ratingMilli ?? 0 });
      }
    }
    out.sort((a, b) => {
      const stateA = this.stateOf(a.provider.id);
      const stateB = this.stateOf(b.provider.id);
      const deadA = stateA.refusals >= DEAD_AFTER_REFUSALS ? 1 : 0;
      const deadB = stateB.refusals >= DEAD_AFTER_REFUSALS ? 1 : 0;
      if (deadA !== deadB) return deadA - deadB;
      if (a.rating !== b.rating) return b.rating - a.rating;
      if (stateA.refusals !== stateB.refusals) return stateA.refusals - stateB.refusals;
      if (stateA.calls !== stateB.calls) return stateA.calls - stateB.calls;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    return out;
  }

  /** One call to one provider: a parsed answer for JSON, the raw body for a stream. */
  async callProvider(candidate: Candidate, req: CRequest): Promise<UpstreamResult> {
    const provider = candidate.provider;
    const base = joinBase(this.envUrl, provider.base);
    let url: string;
    let body: Json;
    if (provider.wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      body = toOpenAI(req, candidate.model);
    } else if (provider.wire === 'anthropic') {
      url = `${base}/v1/messages`;
      body = toAnthropic(req, candidate.model);
    } else {
      const action = req.stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
      url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}${action}`;
      body = toGemini(req);
    }
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      accept: req.stream ? 'text/event-stream' : 'application/json',
      ...authHeaders(provider.auth, provider.key),
    };
    if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
    let res: Response;
    try {
      res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch (err) {
      return { ok: false, status: 0, error: `call to ${provider.id} failed: ${(err as Error).message}` };
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      return {
        ok: false,
        status: res.status,
        error: `${provider.id} answered HTTP ${res.status}${text !== '' ? `: ${text.slice(0, 200)}` : ''}`,
      };
    }
    if (req.stream) {
      if (res.body === null) return { ok: false, status: res.status, error: `${provider.id} answered with no body` };
      return { ok: true, answer: null, stream: res.body };
    }
    let parsed: unknown;
    try {
      parsed = await res.json();
    } catch {
      return { ok: false, status: res.status, error: `${provider.id} answered unparsable JSON` };
    }
    if (!isRecord(parsed)) return { ok: false, status: res.status, error: `${provider.id} answered a non-object` };
    const answer =
      provider.wire === 'openai'
        ? parseOpenAIAnswer(parsed)
        : provider.wire === 'anthropic'
          ? parseAnthropicAnswer(parsed)
          : parseGeminiAnswer(parsed);
    return { ok: true, answer, stream: null };
  }

  /** One client request: route, fail over, serve. */
  async handle(clientWire: ClientWire, body: Json, res: ServerResponse): Promise<void> {
    this.inFlight += 1;
    try {
      const req = clientWire === 'openai' ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
      const model = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
      const index = this.requestCount;
      this.requestCount += 1;
      const requires = this.requirementsFor(index, req);
      const candidates = this.candidatesFor(req, requires);
      let lastError = candidates.length === 0 ? 'no free model meets this request' : 'every provider refused';
      for (const candidate of candidates) {
        const state = this.stateOf(candidate.provider.id);
        for (let attempt = 0; attempt < ATTEMPTS_PER_CANDIDATE; attempt++) {
          state.calls += 1;
          const result = await this.callProvider(candidate, req);
          if (!result.ok) {
            state.refusals += 1;
            lastError = result.error;
            continue;
          }
          state.successes += 1;
          if (req.stream && result.stream !== null) {
            await serveStream(res, clientWire, candidate.provider.wire, result.stream, model);
          } else if (result.answer !== null) {
            serveJson(res, clientWire, result.answer, model);
          } else {
            state.refusals += 1;
            lastError = `${candidate.provider.id} answered with no stream`;
            continue;
          }
          return;
        }
      }
      sendError(res, clientWire, 502, lastError);
    } finally {
      this.inFlight -= 1;
    }
  }
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/** A provider answer as a JSON body in the client's wire, naming the model asked for. */
function serveJson(res: ServerResponse, clientWire: ClientWire, answer: ProviderAnswer, model: string): void {
  const body = clientWire === 'openai' ? openAIJsonAnswer(answer, model) : anthropicJsonAnswer(answer, model);
  const text = JSON.stringify(body);
  res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** A provider stream in the client's wire: piped through untouched on the same wire, converted otherwise. */
async function serveStream(
  res: ServerResponse,
  clientWire: ClientWire,
  providerWire: 'openai' | 'anthropic' | 'gemini',
  stream: ReadableStream<Uint8Array>,
  model: string,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  try {
    if (clientWire === providerWire) {
      for await (const chunk of stream) res.write(chunk);
    } else if (clientWire === 'openai' && providerWire === 'anthropic') {
      for await (const event of anthropicStreamToOpenAI(stream, model)) res.write(event);
    } else if (clientWire === 'openai' && providerWire === 'gemini') {
      for await (const event of geminiStreamToOpenAI(stream, model)) res.write(event);
    } else if (clientWire === 'anthropic' && providerWire === 'openai') {
      for await (const event of openAIStreamToAnthropic(stream, model)) res.write(event);
    } else {
      for await (const event of geminiStreamToAnthropic(stream, model)) res.write(event);
    }
  } catch {
    // The client went away or the stream broke mid-flight; nothing left to serve.
  } finally {
    res.end();
  }
}

/** An error in the client's wire. */
function sendError(res: ServerResponse, clientWire: ClientWire, status: number, message: string): void {
  const body = clientWire === 'openai' ? openAIError(message, 'server_error', null) : anthropicError('api_error', message);
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

/** The request body as a JSON object; null when it is missing, too large, or not JSON. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

async function handlePost(agg: Aggregator, clientWire: ClientWire, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, clientWire, 400, 'the request body is not a JSON object');
    return;
  }
  await agg.handle(clientWire, body, res);
}

function dispatch(agg: Aggregator, req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://localhost');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    const models = agg.servedModels();
    // The caller speaks Anthropic when it authenticates with x-api-key.
    const body = req.headers['x-api-key'] !== undefined ? anthropicModelList(models) : openAIModelList(models);
    const text = JSON.stringify(body);
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    void handlePost(agg, 'openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    void handlePost(agg, 'anthropic', req, res);
    return;
  }
  sendError(res, 'openai', 404, `unknown route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Trial lifecycle
// ---------------------------------------------------------------------------

/** Register with the env, retrying a few times; false when the env never answered. */
async function register(envUrl: string, port: number, catalog: CatalogDoc): Promise<boolean> {
  const root = envUrl.replace(/\/+$/, '');
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${root}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      if (res.ok) return true;
      console.error(`register attempt ${attempt + 1} answered HTTP ${res.status}`);
    } catch (err) {
      console.error(`register attempt ${attempt + 1} failed: ${(err as Error).message}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

/** Poll the env until the trial is done, then wait out any in-flight request and exit 0. */
async function pollWork(envUrl: string, agg: Aggregator, server: Server): Promise<void> {
  const root = envUrl.replace(/\/+$/, '');
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    let done = false;
    try {
      const res = await fetch(`${root}/work`, { signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
      if (res.ok) {
        const body: unknown = await res.json();
        done = isRecord(body) && body.done === true;
      }
    } catch {
      // The env is briefly unreachable; keep polling.
    }
    if (!done) continue;
    while (agg.inFlight > 0) await new Promise((resolve) => setTimeout(resolve, 50));
    server.close();
    process.exit(0);
  }
}

async function main(): Promise<void> {
  const envUrl = process.env.MAKEFUN_ENV_URL ?? '';
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? '';
  const brief = readBrief(promptFile);
  if (brief.error !== null) console.error(`brief: ${brief.error}`);
  const probes = envUrl === '' ? [] : await Promise.all(brief.providers.map((p) => probeProvider(envUrl, p)));
  const catalog = buildCatalog(probes, brief.ratings);
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  const agg = new Aggregator(brief, envUrl, probes, catalog);
  const server = createServer((req, res) => dispatch(agg, req, res));
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  const shutdown = (): void => {
    server.close();
    process.exit(0);
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
  if (envUrl !== '') {
    const registered = await register(envUrl, port, catalog);
    if (!registered) console.error('could not register with the trial env; serving anyway');
    void pollWork(envUrl, agg, server);
  }
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
