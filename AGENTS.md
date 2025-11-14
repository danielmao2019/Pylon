About coding style:

1. I hate defensive programming. Defensive programming is good for production code, but I'm doing research. I want to have my program crash, if it runs out of my expectations. When you code, you should make assumptions about the cases and only write code for the known cases. You must not imagine any potential cases and handle your imagination. If something goes out of my expectation and you handle that case, I can't notice it and hence I am not able to trust any program outcomes. I need to be informed about everything, with hard program abort with the traceback error message. I want those more than a program that runs but produces outcomes out of my control. Examples that should be considered as defensive programming (not strict. In rare cases, these things are indeed needed. I'm just saying you need to be very careful with using these coding styles.):
   1. Try-catch blocks.
   2. Complex if-else handling cases that I don't expect to happen.
   3. Using getattr with a fallback.
   4. Using dict.get with a fallback.
   5. Doing type conversions on objects that I expect to already be of the target type. i.e., doing dict(obj) on an obj that I expect it to be a dict.
   6. Doing .lower() on keys/ids/names.
   7. Using *, in function args.
What you should do instead is extensive assertions, with error messages revealing why the assertion fail. With these, even if you made the wrong assumption the first time you code, we are still safe. We simply gain knowledge about the code and fix those assumptions. But if you hide errors and continue the program failing silently at some point or making use of a fallback dummy number of some "default" and don't have the code fail, I am super worried.
2. You should always make type annotations, especially for the function input args and output.
3. It is bad to make variable aliasing. i.e., you immediately rename a variable to something else, like xxx = yyy. This is common in case when you are making a long sequence of code patches. When you see this, you should stop creating aliasing and rename all subsequent reference to use the original variable name. Note that it is not always the case that xxx = yyy is making an alias. Some times this is just doing initialization. I'm just warning you about this.
4. It is bad to make a function, or a method, that only does trivial things, because these functions/methods only adds unnecessary call stack to the program and does not contribute to code modularity, reusability, and such.
5. Some times there are multiple parts for the code of a function/method and the parts are actually independent. However, you often write code so that the lines for part A is mixed together with lines for part B, making the code hard to understand. You should learn to implement separation of concerns. You either create helper methods, or you make code chunks separated by symbolic comments like # --- or # === or whatever.
6. It's good practice to run black on the files you just made changes within. But never run black on the files you haven't touched - those files are unrelated to the current task and hence should not create any changes.
7. Unless I request explicitly, never create any data classes.
8. Constantly clean up unused variables during implementation and unused imports. For unused function/method args, you need to be careful, because those might be intentionally unused. e.g., a base class defines a method prototype to have a certain arg, but some subclasses use it and other subclasses don't use it.
9. During long sequence of code patches, you often make fallbacks/legacy code. You should never have those. We always move forward.
10. when importing from non-third-party module (module implemented by this repo), you always use absolute import. Sometimes, you need to define repo root relative to __file__ and then you add repo root to sys.path if not there already.

Other coding guidelines

1. When I let you create dash app, you should not do app.run_server, because app.run_server is wrong. You should do app.run. Also, you must always use host 0.0.0.0 and debug=False and make a CLI arg automatically on yourself called --port, with some default value.
