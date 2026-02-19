import os

def cleanup_SAP2000():
    # SAP2000 temp files cannot be deleted while SAP stays open.
    # We disable this cleanup during batch mode.
    return


def cleanup_csv():
    return

if __name__ == "__main__":
    cleanup_SAP2000()
    cleanup_csv(preserve_outputs=False)
