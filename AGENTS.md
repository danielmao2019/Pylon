# Guidelines <!-- omit in toc -->

When I chat with you, sometimes I'm asking a question, sometimes I'm letting you do something. You should distinguish between the two cases. The first section in this doc describes how I expect you to produce an answer to my question, and the second section in this doc describes how I expect you to code.

## 1. About answering my questions

### 1.1. Answer Structure

Your answer must ALWAYS be hierarchical in the level of detail, with the first hierarchy level being the most concise (in one sentence), and the last hierarchy level being the most detailed (a full paragraph or multiple paragraphs containing all detailed information to answer my question). Usually two hierarchy levels would suffice, because in many cases I ask you yes/no questions. Infrequently I ask you questions that can't be answered with yes/no. In such cases you may consider three hierarchy level answers. Do no go beyond three hierarchies.

### 1.2. Building Q/A tree

Usually I will be asking you many questions, and usually when I need your help to analyze a problem, you could be wrong and you need to self-correct. You should maintain a Q/A tree and do depth-first search on the Q/A tree. When I accept your answer and ask a new question, you grow the tree along depth. When I question your previous answer and after deeper discussions you realize that your previous answer was wrong, you need to revert back to that previous node in the tree along depth and retry.

## 2. About coding style

When solving a problem, there are almost always more than 1 feasible solution, and in many cases there are numerous feasible solutions. However, only few of them are in good quality. I know that in most cases you will produce a solution that works, so now I am more worried about the next-level expectations - the quality of the code, and that is exactly why you are reading this section. You should learn from this section what style of code is I consider "good quality code".

### 2.1. Defensive Programming

I hate defensive programming. Defensive programming is good for production code, but I'm doing research. I want to have my program crash, if it runs out of my expectations. When you code, you should make assumptions about the cases and only write code for the known cases. You must not imagine any potential cases and handle your imagination. If something goes out of my expectation and you handle that case, I can't notice it and hence I am not able to trust any program outcomes. I need to be informed about everything, with hard program abort with the traceback error message. I want those more than a program that runs but produces outcomes out of my control. Examples that should be considered as defensive programming (not strict. In rare cases, these things are indeed needed. I'm just saying you need to be very careful with using these coding styles.):
   1. Try-catch blocks.
   2. Complex if-else handling cases that I don't expect to happen.
   3. Using `getattr` with a fallback.
   4. Using `dict.get()` with a fallback.
   5. Doing type conversions on objects that I expect to already be of the target type. i.e., doing `dict(obj)` on an obj that I expect it to be a dict.
   6. Doing `.lower()` on keys/ids/names.
   7. Using `*,` in function args.
What you should do instead is extensive assertions, with error messages revealing why the assertion fail. I said, use assertions, not anything else. If xxx then raise error structure is NOT assertion! With these, even if you made the wrong assumption the first time you code, we are still safe. We simply gain knowledge about the code and fix those assumptions. But if you hide errors and continue the program failing silently at some point or making use of a fallback dummy number of some "default" and don't have the code fail, I am super worried.

Other rules:
1. If you use `zip`, then you must use `strict=True`.

### 2.2. How to write input validation

It is not strictly required to do input validation for all functions/methods and for all args, because sometimes this is not really needed. However, when you do, you must follow the following rules:
1. Location: Input validation must be done at the very beginning of the function/method definition body.
2. Encapsulation: Input validation must be a dedicated area, meaning the followings:
   1. This code section must be the ONLY place that is responsible for input validation. No other places should do input validation, unless it is for a special purpose that's only needed in subsequent logics.
   2. It must start with a line `# Input validations`
   3. It must end with an empty line, before subsequent code in the definition body.
3. Structure:
   1. The lines for each arg should be grouped and put one after another, not mixing the lines.
   2. The order of input validation must follow exactly the order of the input args.
   3. During input validation, ONLY assert statements or dedicated validation module (for complex data structures like camera, COLMAP data, NerfStudio data, etc.) can be used. Introducing variables or transforming input args are strictly prohibited.
   4. During input validation, no `if` conditioning may be used. Each and every statement must be `assert`. You should be using `assert xxx or yyy` to implement `if` conditioning. e.g., for optional args, you should use the structure `assert xxx is None or xxx`, rather than `if xxx is not None: assert xxx`.
4. It is not a strict rule to do input validation for all function args.

### 2.3. How to write input normalization

Input normalization refers to the process, which must be done after all input validations are done, that normalizes different input formats. e.g., a run model function may take a single image, or a list of images, or a single tensor for stacked images; a path may be passed in as `str` type of `pathlib.Path` type; a set of weights may be passed in as a list/tuple of floats, a `numpy.ndarray` instance, a `torch.Tensor` instance, with sum equal to 1, or not equal to 1. Input normalization is, similar to input validation, not strictly needed when the inputs are simple enough, but definitely needs some attention when the inputs are complex.

Rules:
1. Encapsulation: Similar to input validation, input normalization must be a dedicated code section, meaning that
   1. This code section should be the ONLY place that is responsible for input normalization. No other places can do input normalization, unless it is for a special purpose that is only needed in subsequent logics.
   2. It must start with a line `# Input normalizations`.
   3. It must end with an empty line, before any subsequent code in the definition.
2. Structure:
   1. Similar to input validation, input normalization code must be grouped by variables. Do not leave var1 half-normalized and then work on var2 and then go back to var1 to complete the var1 normalization.
   2. Similar to input validation, input normalization code ordering must follow the order of args.

### 2.4. How to write dash apps

Rules:
1. How to define layout:
   1. Must make a folder called `layout`.
   2. Layout must separate the definition of components and styles into different folders/files.
   3. The main API must be exactly `def build_layout(app: Dash) -> None`, where you define `layout` and then assign to the `app` as `app.layout = layout`.
   4. Organize the layout definition hierarchically. Think of the web components as a tree structure. The layout builders should reflect the design of the parent-children relations of the web components. Further, the order of the builders should follow the tree depth-first traversal order.
2. How to define callbacks:
   1. Each callback must contain exactly one `Input`.
   2. Each callback must be defined in a separate file.
   3. Callback functions are not exceptions of the type annotation rules or input validation rules, as defined in other sections of this doc.
   4. Dash callback functions typically need another layer to check for the trigger for mid-to-complex apps. e.g., when there are dynamically created dash components. Be careful with validating if the trigger of the callback is from the actual expected source. If not, then you should use `raise PreventUpdate` to short-circuit the callback. This should be implemented by helper functions of the form `validate_trigger(...) -> None`, called directly by the callbacks. the `validate_trigger` function should have `raise PreventUpdate` statements under various conditions.
   5. Organize callback definition files into sub-folders, when the list gets long.
3. When you make image display, you must never use image encoding.
4. Do not do `app.run_server`, because `app.run_server` is just wrong code. You should do `app.run`. Also, you must always use `host=0.0.0.0`, `port=args.port`, and `debug=False` and make a CLI arg automatically on yourself called `--port`, with some default value.
5. Be careful to the use of multiple callbacks pointing to same `Output` case. Use `allow_duplicate` wisely.

### 2.5. How to write steps and pipelines

Rules:
1. `_init_input_files` and `_init_output_files` must ONLY DEFINE, in a straight-forward way, never raise any error, or do assertions.
2. If any of the following methods are defined, then they must have relative ordering be: `_init_input_files` -> `_init_output_files` -> `check_inputs` -> `check_outputs` -> validation methods -> `run` -> helpers of the `run` method.
3. For COLMAP pipeline, the `_build_colmap_command` must never take any arg other than `self`, and must always return `List[str]`.

### 2.6. How to write type annotations

Rules:
1. You should always make type annotations, especially for the function input args and output.
2. You must never use `object` for type annotation. That's useless.
3. Use `List` from `typing` instead of `list`, `Tuple` from `typing` instead of `tuple`, and `Dict` from `typing` instead of `dict`.
4. Never define types or classes and annotate using defined types or classes.
5. `from __future__ import annotations` should never be used. No new instances of such import shall be created. If any existing instance is spotted, it should be removed immediately regardless or anything else. If the removal creates any problems, the problems should be fixed in alternative ways, which I believe WILL be better than introducing `from __future__ import annotations`.

### 2.7. About user code in this repo

There are a few folders and files that should be considered as user code:
1. The `configs` folder.
2. The `project` folder.
3. The `papers` folder.
4. Many files under repo root, e.g., the `test_*.py` files.

### 2.8. How to write `__init__.py`

Rules:
1. `__init__.py` files must ONLY contain three things: multi-line comment block using `"""`, import statements, and definition of `__all__`.
2. `__init__.py` must never import anything that's not defined under this module.
3. For each sibling file to `__init__.py`, you may import what's defined inside the sibling files, depending on the need of API exposure. You must not import the sibling files themselves. For each sibling folder to `__init__.py`, you may import those as submodule, i.e., import the folders. You must never import anything defined inside that folder. Think of the `__init__.py` files definitions as hierarchical. `__init__.py` must work with and ONLY work with it's immediate children.
4. `__init__.py` is the ONLY place where `__all__` can be defined. i.e., `__all__` must never appear anywhere else.
5. `__init__.py` must go through `isort` (see below), and the items in `__all__` must match exactly the imported items, and in the exact same ordering defined by `isort`.
6. User code folders (as defined above) must never contain ANY `__init__.py` files. This is strict.

### 2.9. How to write import statements

Rules:
1. All imports must be at top of file, except for those intentional lazy imports, which is rare. Any intentional lazy import must be explicitly documented.
2. For user code, items must be imported from modules. The API of the modules are defined by `__init__.py` files.
3. For source code, items must be imported from the exact file, never import from module.
4. Throughout, imports must ALWAYS be absolute imports. I do not expect any single relative import statement.
5. Must run `isort`, but never run `isort` repo-wide. Run `isort` on the files you are working on, ONLY.

Sometimes, you need to add repo root to `sys.path`, in order to use the packages. When doing so, here are the rules:
1. Must define a variable called `REPO_ROOT`.
2. The definition of `REPO_ROOT` must be done in this way:
   1. Must be defined relative to `__file__`.
   2. Must use `pathlib`, not `os.path`.
   3. Must use `parents[x]` where `x` should be determined depending on where the `__file__` is.
3. Addition to `sys.path` must be done in this way:
   1. Must be conditional (avoid duplicates). Do a `if str(REPO_ROOT) not in sys.path`.
   2. Must use `sys.path.append`, rather than `sys.path.insert`.

### 2.10. How to write configs

Configs refer to the `./configs` folder. This folder has a mixture of the following files: a) generator files; b) template files; and c) actual config files. All actual config files have a header (python comment) saying that you should not manually modify this file, and that's exactly what I need to emphasize here. You should only modify the templates or generator files if needed, and then run the generator files to update the corresponding configs.

### 2.11. Code Readability

1. Constantly clean up unused variables during implementation and unused imports. For unused function/method args, you need to be careful, because those might be intentionally unused. e.g., a base class defines a method prototype to have a certain arg, but some subclasses use it and other subclasses don't use it.
2. It's good practice to periodically (in batches) run both `black` and `isort` on the files within which you just made changes. Rules:
   1. Never run `black` or `isort` on the files you haven't touched - those files are unrelated to the current task and hence should not create any changes. Principle: stay within the scope of your task and do not do anything irrelevant.
   2. Never use ANY flags when running `black` or `isort`, especially line length limits.
3. If a function call has more than 3 arguments (positional, or keyword), you must call the function with `func(xxx=xxx, yyy=yyy, ...)`. i.e., ALL args should be called as keyword args. The reason is that with functions with many args, it is easy to mess up with the ordering.
4. The ordering of the args when calling a function, and the order of the input validation assert statements in the function definition, must both follow the same order as the function args.

### 2.12. Code Structure

You are great at making things correct, but not always great at making things human-friendly. This section is dedicated to help you to make your code human-friendly.

#### 2.12.1. A Note on Hierarchies

1. when you talk to a person, you may say something like "meat and fruit", but you less often say things like "meat and apple". because "meat" and "fruit" are on the same hierarchy level, but "apple" is a subclass of "fruit" and hence not on the same hierarchy level as "meat".
2. when you do hierarchical searching in a database of academic papers, you often do three passes - you first filter by title, and then after that, you filter by abstract among the results of filter by title, then you start going into the paper main body, for only the ones you filtered out the abstract. i.e., you first make sure that you stay withing the hierarchy level of paper title, once you are done, you go down one hierarchy level to be more fine-grained and read the abstracts and continue to further filter.
3. when you explain your research idea to other people, you often stay very high level if your audience is not in the same field with you, but you go very deep into the details when you talk to someone in the same field. you can think of this as someone from another field does not have finer hierarchical info built in their head than the very high-level ideas you may describe, so you can only communicate on the highest hierarchical level; for someone in the same field, you both have very deep hierarchical level information stored in your heads, so your communication can happen at very fine-grained levels, using all those technical terminologies and knowledge.
4. when people learn about a system/method/etc., the best way for them to grasp the new thing is to first get an overview. academic papers do method overview section when explaining their proposed method, and books and other long articles have table of contents, professional meetings have agendas laid out before hand.

the list goes on. real human process information in hierarchies. to make your code more human friendly, your code structure should be designed in a hierarchical way. concretely:
1. a functionality that is specific to some hierarchy may not be implemented in a module that is designed to implement more generic functionalities. a generic functionality module may not implement anything that's specific to just one project.
2. a main function may call helper functions, and helper functions may call helper-to-helper functions. but if a main function calls helper-to-helper functions directly, that's a break of the hierarchy structure, even though the code may still be correct, but it's confusing and is considered as bad code quality.

#### 2.12.2. Other Structural Rules

3. It is bad to make variable aliasing. i.e., you immediately rename a variable to something else, like xxx = yyy. This is common in case when you are making a long sequence of code patches. When you see this, you should stop creating aliasing and rename all subsequent reference to use the original variable name. Note that it is not always the case that xxx = yyy is making an alias. Some times this is just doing initialization. I'm just warning you about this.
4. It is bad to make a function, or a method, that only does trivial things, because these functions/methods only adds unnecessary call stack to the program and does not contribute to code modularity, reusability, and such.
5. Some times there are multiple parts for the code of a function/method and the parts are actually independent. However, you often write code so that the lines for part A is mixed together with lines for part B, making the code hard to understand. You should learn to implement separation of concerns. You either create helper methods, or you make code chunks separated by symbolic comments like `# ---` or `# ===` or whatever.

### 2.13. First-Principle Rule

1. During long sequence of code patches, you often make fallbacks/legacy code. You should never have those. We always move forward.

### 2.14. Others

1. Unless I request explicitly, never create any data classes.

## 3. About coding workflow

Rule: never do any `git` commit that is not read-only, e.g., `git add`, `git commit`, `git push`, `git pull`, `git rebase`, `git stash`, etc., unless the user tells you to do so explicitly.

Suggestion: git diff is always your friend. You should review what you did constantly. Use git diff wisely, e.g., `git diff --staged`, `git diff --cached`, `git diff --stat`, etc.

I prove your changes by doing `git add` and `git commit` manually myself. If you notice that some of your changes is gone from `git diff`, you should think if it's a failure of changes applying to the files, or if it's because of my add and commit. You should think of my add and commit as (weak) "approval" of your changes - subject to the additional follow-up prompts I give you to make follow-up code changes. Pay more attention to what's left un-added or un-committed and the follow-up prompts I give. Those are usually strong indications of the remaining work to be done.
