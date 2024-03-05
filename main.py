# To activate the environment variable file 
from dotenv import load_dotenv
import os 
import pandas as pd # To load csv and read
from llama_index.core.query_engine import PandasQueryEngine # Allows us to ask specific question from the data source / other query engine avail
from prompts import new_prompt, instruction_str, context# A template / give user a human type of responce / look into file
# Importing more tools below 
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI # The one shown now <-- work. before i had from llama_index.core.llms import OpenAI which did not work said it had import errors and such
from pdf import canada_engine # adding another tool

# Call the load_dotenv file which will look for the presence of a .env file and load the environment variables 
load_dotenv()

population_path = os.path.join("data", "population.csv") # Specify the path of our data
population_df = pd.read_csv(population_path) # Load the csv

#print(population_df.head())

population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str) # Verbose allows us to see the thoughts. / All this is doing is wrapping on top of the 
                                                                            # pandas df and giving us an interface via like a retrieval generation augmented system to ask question.

population_query_engine.update_prompts({"pandas_prompt": new_prompt})

#population_query_engine.query("What is the population of canada") # Now we can query directly from the source after passing in the prompt / instruction string
                                                                  # NOTE: Now our agent can use this tool to read the output and parse it/with other it needs to give us a more human readable/response

# Now provide a list of what tools we have access too 
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information of the world population and demographics"
        )
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="this gives detailed informatino about canada the country"
        )
    )
]

# Now we will set up an agent that can have access to the tools above and query from the data
llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context) # Tell the reactagent what tools to use and context can be used to give infroamtion or context beforehand on what to do/etc
                                                                        # to specify a context string we can go to prompts and type it there

# Now we can create a while loop to continually use the agent and ask it diff prompts and then the agent can utilize thetools and give response
while(prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

# https://www.youtube.com/watch?v=ul0QsodYct4 
    # stopped at the running the ai model

# agent based-ai we are able to pass functions, dataset, tools to an llm to let it reasonm and allow us human to have a little more control
# but to let the agent to make the decison on what tool to use and what it actually it needs to do
    
# look into the llama-index memory to remember chat history and stuff 