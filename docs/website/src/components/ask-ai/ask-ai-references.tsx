'use client';

import { ExternalLink } from 'lucide-react';

import type { AskAIReference } from './use-ask-ai-chat';

/**
 * Derive a human-readable label for a source link: prefer the
 * Inkeep-provided title, fall back to the URL's hostname (stripped of
 * a leading `www.`), and finally the raw URL.
 */
function referenceLabel(reference: AskAIReference): string {
  const title = reference.title?.trim();
  if (title) return title;
  try {
    return new URL(reference.url).hostname.replace(/^www\./, '');
  } catch {
    return reference.url;
  }
}

/**
 * "Sources" footer shown beneath an assistant answer. Lists the
 * citations Inkeep returned via the `provideLinks` tool so readers can
 * jump to the underlying docs. Renders nothing when there are no
 * references (e.g. while streaming, or for answers with no sources).
 */
export function AskAIReferences({ references }: { references: AskAIReference[] }) {
  if (references.length === 0) return null;
  return (
    <div className="mt-3 border-t border-fd-border/60 pt-2">
      <p className="mb-1.5 text-xs font-medium text-fd-muted-foreground">Sources</p>
      <ol className="flex flex-col gap-1">
        {references.map((reference, index) => (
          <li key={reference.url} className="flex items-start gap-1.5 text-xs">
            <span className="mt-px shrink-0 tabular-nums text-fd-muted-foreground">
              {index + 1}.
            </span>
            <a
              href={reference.url}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex min-w-0 items-center gap-1 text-fd-primary underline underline-offset-2 hover:text-fd-primary/80"
            >
              <span className="truncate">{referenceLabel(reference)}</span>
              <ExternalLink className="size-3 shrink-0" aria-hidden="true" />
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}
