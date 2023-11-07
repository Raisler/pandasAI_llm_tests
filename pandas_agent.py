from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

csv_file_path = "iris.csv"
df = pd.read_csv(csv_file_path)
df.head(3)

llm = Ollama(
    model="mistral", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)
agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True,max_execution_time=20, 
                                      max_iterations=1, handle_parsing_errors=True)

response = agent.run('bring the number of rows')