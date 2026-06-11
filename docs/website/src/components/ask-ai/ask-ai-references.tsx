'use client';

import { ExternalLink, FileText } from 'lucide-react';

import type { AskAIReference } from './use-ask-ai-chat';

/**
 * Path segments that read as acronyms in the docs nav and should be
 * fully uppercased in a breadcrumb (rather than title-cased to "Ai",
 * "Cli", …). Lowercase keys; matched case-insensitively.
 */
const SECTION_ACRONYMS = new Set([
  'ai',
  'api',
  'cli',
  'kv',
  'llm',
  'ocr',
  'p2p',
  'qvac',
  'rag',
  'sdk',
  'tts',
]);

/**
 * Turn a URL slug segment into a display label: split on hyphens and
 * either uppercase known acronyms or capitalize the first letter.
 * e.g. `ai-capabilities` -> "AI Capabilities", `fine-tuning` -> "Fine Tuning".
 */
function formatSegment(segment: string): string {
  return segment
    .split('-')
    .filter(Boolean)
    .map((word) =>
      SECTION_ACRONYMS.has(word.toLowerCase())
        ? word.toUpperCase()
        : word.charAt(0).toUpperCase() + word.slice(1),
    )
    .join(' ');
}

/**
 * Derive the card's title: prefer the Inkeep-provided title, fall back
 * to the formatted final URL slug (e.g. `video-generation` ->
 * "Video Generation"), and finally the hostname / raw URL.
 */
function referenceTitle(reference: AskAIReference): string {
  const title = reference.title?.trim();
  if (title) return title;
  try {
    const url = new URL(reference.url);
    const segments = url.pathname.split('/').filter(Boolean);
    const last = segments[segments.length - 1];
    if (last) return formatSegment(last);
    return url.hostname.replace(/^www\./, '');
  } catch {
    return reference.url;
  }
}

/**
 * Derive the card's breadcrumb (the section the page lives in) straight
 * from the URL path — Inkeep's `provideLinks` payload doesn't carry a
 * reliable section field, but our docs URLs encode it
 * (`/docs/ai-capabilities/video-generation` -> "AI Capabilities"). We
 * drop a leading `docs` prefix and the final page slug; whatever section
 * segments remain become the breadcrumb. Returns `null` when nothing
 * meaningful can be derived (so the card just shows the title).
 */
function referenceBreadcrumb(reference: AskAIReference): string | null {
  try {
    const url = new URL(reference.url);
    const segments = url.pathname.split('/').filter(Boolean);
    const isDocs = segments[0] === 'docs';
    const trimmed = isDocs ? segments.slice(1) : segments;
    const sections = trimmed.slice(0, -1);
    if (sections.length > 0) return sections.map(formatSegment).join(' / ');
    if (isDocs) return 'Docs';
    return url.hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

/**
 * "Sources" footer shown beneath an assistant answer. Renders the
 * citations Inkeep returned via the `provideLinks` tool as compact,
 * fully-clickable cards (icon + breadcrumb + title) so readers can jump
 * to the underlying docs — mirroring the legacy Inkeep widget's source
 * cards. Renders nothing when there are no references (e.g. while
 * streaming, or for answers with no sources).
 */
export function AskAIReferences({ references }: { references: AskAIReference[] }) {
  if (references.length === 0) return null;
  return (
    <div className="mt-3 border-t border-fd-border/60 pt-2.5">
      <p className="mb-1.5 text-xs font-medium text-fd-muted-foreground">Sources</p>
      <ul className="flex flex-col gap-1.5">
        {references.map((reference) => {
          const breadcrumb = referenceBreadcrumb(reference);
          return (
            <li key={reference.url}>
              <a
                href={reference.url}
                target="_blank"
                rel="noreferrer noopener"
                className="group flex items-center gap-2.5 rounded-lg border border-fd-border bg-fd-card px-2.5 py-2 transition-colors hover:bg-fd-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fd-ring"
              >
                <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-md border border-fd-border bg-fd-popover text-fd-muted-foreground transition-colors group-hover:text-fd-primary">
                  <FileText className="size-3.5" aria-hidden="true" />
                </span>
                <span className="flex min-w-0 flex-col">
                  {breadcrumb ? (
                    <span className="truncate text-[0.7rem] leading-tight text-fd-muted-foreground">
                      {breadcrumb}
                    </span>
                  ) : null}
                  <span className="truncate text-xs font-medium text-fd-popover-foreground">
                    {referenceTitle(reference)}
                  </span>
                </span>
                <ExternalLink
                  className="ml-auto size-3 shrink-0 text-fd-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                  aria-hidden="true"
                />
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
