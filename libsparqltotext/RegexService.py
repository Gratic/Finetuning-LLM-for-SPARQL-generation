import re

class RegexService():
    def __init__(self, args) -> None:
        if args.verbose:
            print("Starting execution.")
            print("Compiling regex... ", end="")
        
        self.pattern = re.compile(r'\"[A-Z].*\"', flags=0)
        
        if args.verbose:
            print("Done.")
    
    def extract_prompts(self, generated_text):
        return self.pattern.findall(generated_text)