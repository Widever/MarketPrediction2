import importlib
import traceback
import command as cmd

while True:
    try:
        print("Enter to continue...")
        input_ = input()
        if input_ == "stop":
            break

        importlib.reload(cmd)
        cmd.run()
    except Exception as e:
        traceback.print_exception(e)
