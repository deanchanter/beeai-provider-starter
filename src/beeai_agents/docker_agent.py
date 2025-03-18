from pydantic import Field,BaseModel
from typing import Literal
import traceback
from pydantic import BaseModel, ValidationError
from beeai_framework.backend.chat import ChatModel, ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils.templates import PromptTemplate
from beeai_framework.workflows.workflow import Workflow, WorkflowError
from dotenv import load_dotenv
import os
import asyncio
load_dotenv(override=True)

# Workflow State
class DockerAgentState(BaseModel):
    readme: str | None = None
    dockerfile: str | None = None
    file_path: str

# PromptTemplate Input Schemas
class READMEInput(BaseModel):
    readme: str

class DockerfileInput(BaseModel):
    dockerfile: str

# Structured output Schemas
class DockerfileOutput(BaseModel):
    dockerfile: str = Field(description="The dockerfile for the application")

class DockerScore(BaseModel):
    score: Literal["yes", "no"] = Field(description="The score of the dockerfile")


# Create a ChatModel to
model = ChatModel.from_name(
    "watsonx:meta-llama/llama-3-3-70b-instruct",
    options={
        "project_id": os.getenv('WATSONX_PROJECT_ID',None),
        "api_key": os.getenv('WATSONX_API_KEY',None),
        "api_base": os.getenv('WATSONX_API_URL',None)
    },
)

# Prompt Templates
readme_template = PromptTemplate(
    schema=READMEInput,
    template="""<|start_of_role|>system<|end_of_role|>
You are a helpful assistant who is an expert in building container applications.

Given the following README file: {{readme}}, create a multi-stage Dockerfile for the source code found in the repository.

Please be sure to copy ALL source code files from the root folder into the build stage.
Where you find strings like `project_version` in the README, substitute specific version information.
If you do not know version information, use `0.0.1`.

Expose any needed ports for the application.

Ensure that any needed environment variables are passed into the build and runtime stages of the dockerfile.

Source all container images from the registry {{readme}}

<|end_of_text|>""",
)

dockerfile_checker_template = PromptTemplate(
    schema=DockerfileInput,
    template="""<|start_of_role|>system<|end_of_role|>
Your task is to check that application Dockerfile meets the requirement of the user's README file.
Return your binary `yes` or `no` score in a string, with the single key `score`.
<|end_of_text|>"""
)
def read_markdown_file(state: DockerAgentState) -> str:
    """
    Reads the contents of a markdown file.
    """
    print("Step: ", "Injesting README")
    try:
        with open(state.file_path, 'r') as file:
            contents = file.read()
            state.readme = contents
            return "generate_dockerfile"
    except FileNotFoundError:
        print(f"File not found: {state.file_path}")
        return None

async def generate_dockerfile(state: DockerAgentState) -> str:
    print("Step: ", "Constructing Dockerfile")
    prompt = readme_template.render(READMEInput(readme=state.readme))
    response: ChatModelStructureOutput = await model.create_structure(
        {
            "schema": DockerfileOutput,
            "messages": [UserMessage(prompt)],
        }
    )
    # Run search and store results in state
    state.dockerfile = response.object["dockerfile"]
    return "check_dockerfile"

async def check_dockerfile(state: DockerAgentState) -> str:
    print("Step: ", "Check Dockerfile")
    # Generate answer based on question and search results from previous step.
    prompt = dockerfile_checker_template.render(DockerfileInput(dockerfile=state.dockerfile))
    output: ChatModelStructureOutput = await model.create_structure(
        {
            "messages": [UserMessage(prompt)],
            "schema": DockerScore,
        }
    )

    # Store answer in state
    score = output.object["score"]
    if score == "yes":
        return Workflow.END
    else:
        return "generate_dockerfile"
    
async def main():
    try:
        # Define the structure of the workflow graph
        docker_agent_workflow = Workflow(schema=DockerAgentState, name="DockerAgent")
        docker_agent_workflow.add_step("read_markdown_file", read_markdown_file)
        docker_agent_workflow.add_step("generate_dockerfile", generate_dockerfile)
        docker_agent_workflow.add_step("check_dockerfile", check_dockerfile)

        readme = '/Users/deanchanter/Documents/GitHub/beeai-provider-starter/src/beeai_agents/EXAMPLE_README.MD'
        # Execute the workflow
        response = await docker_agent_workflow.run(
            DockerAgentState(file_path=readme)
        )

        print("*****")
        print("Readme:\n\n", response.state.readme)
        print("DockerFile:\n\n", response.state.dockerfile)

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()
    return response.state.dockerfile

if __name__ == "__main__":
    asyncio.run(main())