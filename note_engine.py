# Create a function can be more complex. But a function that our model can execute 
from llama_index.core.tools import FunctionTool # Tells llama index that we have function that the model can use/tool
import os 

note_file  = os.path.join("data", "notes.txt")

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([note + "\n"]) # Pass a list, appends and goes to new line after

    return "note saved" # Can really return anything. In this case our model can actually read and use it as an indicator if our file worked/saved/etc 

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user"
)
    # Now its important when passing things here because this can help the model decide what tool to use. So being more specific or context can help the 
    # model understand what this does. 
