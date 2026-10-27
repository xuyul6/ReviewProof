export interface SSEMessage {
  event: string;
  data: unknown;
}

// Parses a server-sent event stream whose chunks may split an event (or its blank-line
// separator) anywhere. Multi-line data is joined with "\n" before JSON.parse.
export async function* parseSSE(chunks: AsyncIterable<string>): AsyncGenerator<SSEMessage> {
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += chunk.replace(/\r\n?/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const message = parseBlock(block);
      if (message) yield message;
    }
  }
  const last = parseBlock(buffer);
  if (last) yield last;
}

function parseBlock(block: string): SSEMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (data.length === 0) return null;
  return { event, data: JSON.parse(data.join("\n")) };
}
