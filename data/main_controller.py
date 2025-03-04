import importlib
import sys
import traceback
import command as cmd

def flush_input():
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        import sys, termios    #for linux/unix
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


while True:
    try:
        flush_input()
        input_ = input("Enter to run command.py ...")
        if input_ == "stop":
            break

        importlib.reload(cmd)
        cmd.run()
    except Exception as e:
        traceback.print_exception(e)
