import asyncio

from acp.server.highlevel import Server, Context
from beeai_sdk.providers.agent import run_agent_provider
from beeai_sdk.schemas.metadata import UiDefinition, UiType
from beeai_sdk.schemas.text import TextInput, TextOutput

from beeai_agents.configuration import Configuration
from beeai_agents.docker_agent import docker_agent


async def run():
    server = Server("beeai-agents")

    @server.agent(
        name="devops_agent",
        description="Make a docker file from a README",
        input=TextInput,
        output=TextOutput,
        ui=UiDefinition(type=UiType.hands_off, userGreeting="Please provide a README file"),
    )
    async def devops_agent(input: TextInput, ctx: Context) -> TextOutput:
        """TODO: Your implementation goes here."""
        results = await docker_agent(input.text)
        # results = "hello world"
        return TextOutput(text=results)

    await run_agent_provider(server)


def main():
    asyncio.run(run())
