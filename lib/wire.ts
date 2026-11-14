// The three wires: OpenAI Chat Completions, Anthropic Messages, Gemini generateContent.
//
// The client speaks OpenAI or Anthropic; a provider speaks OpenAI, Anthropic or Gemini.
// Every client request is first read into one canonical shape (system text, messages of
// content blocks, tools, caps), then written out in the provider's wire; every provider
// answer is read into one shape (text, usage, stop reason, tool calls) and written out
// in the client's wire, as JSON or as server-sent events.
//
// The provider echoes what it parsed — system text, user text, tools, max_tokens, stop
// sequences, image, JSON mode — and the client compares that echo with what it sent, so
// the translations preserve those fields exactly: the same strings, the same tool names
// in the same order, the same base64 bytes for images, JSON mode only on a wire that
// can carry it (Anthropic's cannot). Everything here is a pure function of its inputs:
// no clock, no randomness, fixed timestamps in the model lists.

import { isRecord, type Json } from './brief.ts';

/** Fixed epoch second for every listed model's `created`. */
const FIXED_CREATED = 1767225600;
/** Fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

export interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** GET /v1/models, OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models, Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's envelope. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's envelope. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Canonical request
// ---------------------------------------------------------------------------

export type CImage = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; url: string };

export type CBlock =
  | { kind: 'text'; text: string }
  | { kind: 'image'; image: CImage }
  | { kind: 'toolUse'; id: string; name: string; input: Json }
  | { kind: 'toolResult'; toolUseId: string; content: string };

export interface CMessage {
  role: 'user' | 'assistant' | 'tool';
  blocks: CBlock[];
  /** The tool_call_id of a tool-role message (OpenAI's spelling). */
  toolCallId?: string;
}

export interface CTool {
  name: string;
  description: string;
  parameters: Json;
}

export type CToolChoice = 'auto' | 'required' | 'none' | { name: string };

export interface CRequest {
  system: string;
  messages: CMessage[];
  tools: CTool[];
  toolChoice: CToolChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function strList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) {
    if (isRecord(b) && b.type === 'text' && typeof b.text === 'string') parts.push(b.text);
  }
  return parts.join('\n');
}

/** A data URL or plain URL as a canonical image; null when it is neither. */
function imageOfUrl(url: unknown): CImage | null {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'base64', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2] };
  return { kind: 'url', url };
}

/** A tool call's arguments as an object; {} when they are not a JSON object. */
function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: empty input.
    }
  }
  return {};
}

/** A tool_result block's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** OpenAI content parts as canonical blocks. */
function openAIBlocks(content: unknown): CBlock[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: CBlock[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ kind: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url)) {
      const image = imageOfUrl(part.image_url.url);
      if (image !== null) blocks.push({ kind: 'image', image });
    } else if (part.type === 'input_audio') {
      // No wire in this trial carries audio; the echo does not cover it either.
    }
  }
  return blocks;
}

/** Anthropic content blocks as canonical blocks. */
function anthropicBlocks(content: unknown): CBlock[] {
  if (typeof content === 'string') return content === '' ? [] : [{ kind: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const blocks: CBlock[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') blocks.push({ kind: 'text', text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        blocks.push({
          kind: 'image',
          image: { kind: 'base64', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data },
        });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        blocks.push({ kind: 'image', image: { kind: 'url', url: source.url } });
      }
    } else if (block.type === 'tool_use') {
      blocks.push({
        kind: 'toolUse',
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        input: isRecord(block.input) ? block.input : {},
      });
    } else if (block.type === 'tool_result') {
      blocks.push({
        kind: 'toolResult',
        toolUseId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: toolResultText(block.content),
      });
    }
  }
  return blocks;
}

/** An OpenAI Chat Completions body as a canonical request. */
export function canonicalFromOpenAI(body: Json): CRequest {
  const systemParts: string[] = [];
  const messages: CMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        blocks: [{ kind: 'toolResult', toolUseId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: textOf(raw.content) }],
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : undefined,
      });
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks = openAIBlocks(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          kind: 'toolUse',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', blocks });
      continue;
    }
    messages.push({ role: 'user', blocks: openAIBlocks(raw.content) });
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'required') toolChoice = 'required';
  else if (choice === 'none') toolChoice = 'none';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a canonical request. */
export function canonicalFromAnthropic(body: Json): CRequest {
  const messages: CMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') messages.push({ role: 'assistant', blocks: anthropicBlocks(raw.content) });
    else messages.push({ role: 'user', blocks: anthropicBlocks(raw.content) });
  }
  const tools: CTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    system: textOf(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Canonical request -> provider wire
// ---------------------------------------------------------------------------

/** Canonical blocks as OpenAI content parts. */
function openAIParts(blocks: CBlock[]): Json[] {
  const parts: Json[] = [];
  for (const block of blocks) {
    if (block.kind === 'text') parts.push({ type: 'text', text: block.text });
    else if (block.kind === 'image') {
      if (block.image.kind === 'base64') {
        parts.push({ type: 'image_url', image_url: { url: `data:${block.image.mime};base64,${block.image.data}` } });
      } else {
        parts.push({ type: 'image_url', image_url: { url: block.image.url } });
      }
    }
  }
  return parts;
}

/** A canonical request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(req: CRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const msg of req.messages) {
    if (msg.role === 'tool') {
      const result = msg.blocks.find((b): b is Extract<CBlock, { kind: 'toolResult' }> => b.kind === 'toolResult');
      messages.push({
        role: 'tool',
        tool_call_id: msg.toolCallId ?? result?.toolUseId ?? '',
        content: result?.content ?? '',
      });
      continue;
    }
    if (msg.role === 'assistant') {
      const toolCalls: Json[] = [];
      for (const block of msg.blocks) {
        if (block.kind !== 'toolUse') continue;
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: JSON.stringify(block.input) },
        });
      }
      const text = msg.blocks.flatMap((b) => (b.kind === 'text' ? [b.text] : [])).join('\n');
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts = openAIParts(msg.blocks);
    const results = msg.blocks.filter((b): b is Extract<CBlock, { kind: 'toolResult' }> => b.kind === 'toolResult');
    for (const result of results) {
      messages.push({ role: 'tool', tool_call_id: result.toolUseId, content: result.content });
    }
    if (parts.length === 1 && parts[0]?.type === 'text') messages.push({ role: 'user', content: parts[0].text });
    else if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (req.toolChoice === 'required') out.tool_choice = 'required';
  else if (req.toolChoice === 'none') out.tool_choice = 'none';
  else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** Canonical blocks as Anthropic content blocks. */
function anthropicBlocksOut(blocks: CBlock[]): Json[] {
  const out: Json[] = [];
  for (const block of blocks) {
    if (block.kind === 'text') out.push({ type: 'text', text: block.text });
    else if (block.kind === 'image') {
      if (block.image.kind === 'base64') {
        out.push({ type: 'image', source: { type: 'base64', media_type: block.image.mime, data: block.image.data } });
      } else {
        out.push({ type: 'image', source: { type: 'url', url: block.image.url } });
      }
    } else if (block.kind === 'toolUse') {
      out.push({ type: 'tool_use', id: block.id, name: block.name, input: block.input });
    } else if (block.kind === 'toolResult') {
      out.push({ type: 'tool_result', tool_use_id: block.toolUseId, content: block.content });
    }
  }
  return out;
}

/** A canonical request as an Anthropic Messages body naming `model`. */
export function toAnthropic(req: CRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'tool') {
      const result = msg.blocks.find((b): b is Extract<CBlock, { kind: 'toolResult' }> => b.kind === 'toolResult');
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: msg.toolCallId ?? result?.toolUseId ?? '', content: result?.content ?? '' }],
      });
      continue;
    }
    const blocks = anthropicBlocksOut(msg.blocks);
    if (blocks.length === 0) continue;
    // A lone text block may stay a string; the echo reads both the same way.
    const singleText = blocks.length === 1 && blocks[0]?.type === 'text' ? blocks[0].text : null;
    messages.push({ role: msg.role, content: singleText !== null ? singleText : blocks });
  }
  const out: Json = { model, messages };
  if (req.system !== '') out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (req.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stream) out.stream = true;
  return out;
}

/** Canonical blocks as Gemini content parts. `names` maps tool call ids to function names. */
function geminiParts(blocks: CBlock[], names: Map<string, string>): Json[] {
  const parts: Json[] = [];
  for (const block of blocks) {
    if (block.kind === 'text') parts.push({ text: block.text });
    else if (block.kind === 'image') {
      if (block.image.kind === 'base64') {
        parts.push({ inlineData: { mimeType: block.image.mime, data: block.image.data } });
      } else {
        parts.push({ fileData: { fileUri: block.image.url } });
      }
    } else if (block.kind === 'toolUse') {
      names.set(block.id, block.name);
      parts.push({ functionCall: { name: block.name, args: block.input } });
    } else if (block.kind === 'toolResult') {
      parts.push({
        functionResponse: { name: names.get(block.toolUseId) ?? block.toolUseId, response: { result: block.content } },
      });
    }
  }
  return parts;
}

/** A canonical request as a Gemini generateContent body. The model travels in the URL. */
export function toGemini(req: CRequest): Json {
  const names = new Map<string, string>();
  const contents: Json[] = [];
  for (const msg of req.messages) {
    const parts = geminiParts(msg.blocks, names);
    if (parts.length === 0) continue;
    // Tool results ride as user-role content; assistant turns are the model's.
    contents.push({ role: msg.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (req.toolChoice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (req.toolChoice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (req.toolChoice !== null && typeof req.toolChoice === 'object') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [req.toolChoice.name] } };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers -> client wire (JSON)
// ---------------------------------------------------------------------------

export interface ToolCallAnswer {
  id: string;
  name: string;
  args: Json;
}

export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
  /** The provider's own stop reason (stop, end_turn, STOP, ...). */
  stop: string;
  toolCalls: ToolCallAnswer[];
}

function zeroOr(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** An OpenAI Chat Completions answer as a provider answer. */
export function parseOpenAIAnswer(body: Json): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: ToolCallAnswer[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args: argsObject(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: { input: zeroOr(usage.prompt_tokens), output: zeroOr(usage.completion_tokens) },
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
    toolCalls,
  };
}

/** An Anthropic Messages answer as a provider answer. */
export function parseAnthropicAnswer(body: Json): ProviderAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ToolCallAnswer[] = [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        input: isRecord(block.input) ? block.input : {},
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: { input: zeroOr(usage.input_tokens), output: zeroOr(usage.output_tokens) },
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
    toolCalls,
  };
}

/** A Gemini generateContent answer as a provider answer. */
export function parseGeminiAnswer(body: Json): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ToolCallAnswer[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        args: isRecord(part.functionCall.args) ? part.functionCall.args : {},
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    usage: { input: zeroOr(usage.promptTokenCount), output: zeroOr(usage.candidatesTokenCount) },
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
    toolCalls,
  };
}

/** A provider stop reason as an OpenAI finish_reason. */
export function toOpenAIStop(reason: string | null): string {
  switch (reason) {
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 'length';
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 'tool_calls';
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
      return 'content_filter';
    default:
      return 'stop';
  }
}

/** A provider stop reason as an Anthropic stop_reason. */
export function toAnthropicStop(reason: string | null): string {
  switch (reason) {
    case 'length':
    case 'max_tokens':
    case 'MAX_TOKENS':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
    case 'tool_use':
      return 'tool_use';
    case 'content_filter':
    case 'refusal':
    case 'SAFETY':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

/** A provider answer as an OpenAI Chat Completions object naming the requested model. */
export function openAIJsonAnswer(answer: ProviderAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toOpenAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A provider answer as an Anthropic Messages object naming the requested model. */
export function anthropicJsonAnswer(answer: ProviderAnswer, model: string): Json {
  const content: Json[] = [{ type: 'text', text: answer.text }];
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_aggregate',
      name: call.name,
      input: call.args,
    });
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toAnthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events
// ---------------------------------------------------------------------------

/** One SSE frame: `event: <name>` when named, the JSON data line, the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

export interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrame(frame: string): SseFrame | null {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  if (data.length === 0) return null;
  return { event, data: data.join('\n') };
}

/** A byte stream as SSE frames, split on blank lines, CRLF normalized. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        if (frame !== null) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const frame = parseSseFrame(buffer);
      if (frame !== null) yield frame;
    }
  } finally {
    reader.releaseLock();
  }
}

function parseJsonFrame(data: string): Json | null {
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/**
 * An OpenAI event stream as Anthropic's event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the
 * stop reason and output tokens, then message_stop. The stop reason is held until [DONE]:
 * OpenAI sends usage after the finish chunk, and message_delta carries the output tokens.
 */
export async function* openAIStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const stopOpenBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...stopOpenBlocks(),
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      yield* startEvents();
      yield* finishEvents(toAnthropicStop(pendingStop));
      continue;
    }
    const chunk = parseJsonFrame(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = zeroOr(chunk.usage.prompt_tokens);
      outputTokens = zeroOr(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    yield* startEvents();
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index: textIndex,
          content_block: { type: 'text', text: '' },
        });
      }
      yield sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: textIndex,
        delta: { type: 'text_delta', text: delta.content },
      });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index,
          delta: { type: 'input_json_delta', partial_json: fn.arguments },
        });
      }
    }
    if (typeof choice.finish_reason === 'string') pendingStop = choice.finish_reason;
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents(toAnthropicStop(pendingStop));
  }
}

/**
 * An Anthropic event stream as OpenAI chunks: a role chunk, content deltas, tool-call
 * deltas, a finish chunk, a usage chunk carrying both counts, then [DONE].
 */
export async function* anthropicStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolIndexes = new Map<number, number>();
  let nextTool = 0;

  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      chunk({}, toOpenAIStop(stopReason)),
      sseEvent(null, {
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const data = parseJsonFrame(frame.data);
    if (data === null) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = zeroOr(data.message.usage.input_tokens);
        yield* startEvents();
        break;
      }
      case 'content_block_start': {
        yield* startEvents();
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0 && !toolIndexes.has(index)) {
          const toolIndex = nextTool;
          nextTool += 1;
          toolIndexes.set(index, toolIndex);
          yield chunk(
            {
              tool_calls: [
                {
                  index: toolIndex,
                  id: typeof block.id === 'string' ? block.id : `call_${toolIndex}`,
                  type: 'function',
                  function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
                },
              ],
            },
            null,
          );
        }
        break;
      }
      case 'content_block_delta': {
        yield* startEvents();
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          yield chunk({ content: delta.text }, null);
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const toolIndex = toolIndexes.get(index);
          if (toolIndex !== undefined) {
            yield chunk({ tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null);
          }
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
        if (isRecord(data.usage)) outputTokens = zeroOr(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        yield* startEvents();
        yield* finishEvents();
        break;
      }
      default:
        break;
    }
  }
  if (!finished) {
    yield* startEvents();
    yield* finishEvents();
  }
}

/**
 * A Gemini event stream as OpenAI chunks. Gemini's stream carries no [DONE] marker and
 * its usage only in the last frame, so the finish chunk, the usage chunk and [DONE] are
 * emitted when the stream ends.
 */
export async function* geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;

  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-stream',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      chunk({}, toOpenAIStop(stopReason)),
      sseEvent(null, {
        id: 'chatcmpl-stream',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
      }),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    const data = parseJsonFrame(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    yield* startEvents();
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield chunk({ content: part.text }, null);
      }
    }
    if (typeof candidate.finishReason === 'string') stopReason = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = zeroOr(data.usageMetadata.promptTokenCount);
      outputTokens = zeroOr(data.usageMetadata.candidatesTokenCount);
    }
  }
  yield* startEvents();
  yield* finishEvents();
}

/**
 * A Gemini event stream as Anthropic's event sequence. The input count is only known in
 * the last frame, after message_start has gone out, so message_start carries 0; the
 * stream's check reads only the output count from message_delta.
 */
export async function* geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let textOpen = false;
  let outputTokens = 0;
  let stopReason: string | null = null;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'text', text: '' },
      }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      textOpen = false;
    }
    out.push(
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: toAnthropicStop(stopReason), stop_sequence: null },
        usage: { output_tokens: outputTokens },
      }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    const data = parseJsonFrame(frame.data);
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const events = startEvents();
    if (events.length > 0) {
      textOpen = true;
      yield* events;
    }
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield sseEvent('content_block_delta', {
          type: 'content_block_delta',
          index: 0,
          delta: { type: 'text_delta', text: part.text },
        });
      }
    }
    if (typeof candidate.finishReason === 'string') stopReason = candidate.finishReason;
    if (isRecord(data.usageMetadata)) outputTokens = zeroOr(data.usageMetadata.candidatesTokenCount);
  }
  yield* startEvents();
  yield* finishEvents();
}
