import streamlit as st
import threading
import time
import queue

done = False


def worker():
    counter = 0
    while not done:
        time.sleep(1)
        counter += 1
        print(counter)
        
       
threading.Thread(target=worker, daemon=True).start()



print(st.__version__)
input("Press enter to quit")
done = True