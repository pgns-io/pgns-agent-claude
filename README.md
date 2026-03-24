# pgns-agent-claude

Claude Agent SDK adapter for [pgns-agent](https://pypi.org/project/pgns-agent/). Wrap a Claude agent in a production-ready A2A server.

## Installation

```bash
pip install pgns-agent-claude
```

## Quick Start

```python
from claude_agent_sdk import Agent
from pgns_agent import AgentServer
from pgns_agent_claude import ClaudeAdapter

claude_agent = Agent(model="claude-sonnet-4-5-20250514")

server = AgentServer("my-agent", "A Claude-powered agent")
server.use(ClaudeAdapter(claude_agent))
server.listen(3000)
```

## License

Apache-2.0
