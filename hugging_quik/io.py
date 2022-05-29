from pathlib import Path
import json


def json_write(serve_path: Path, filename: str, data: str):
    """Simple function to write json output to the appropriate
    directory and filename.

    Args:
        serve_path (Path): Directory for model archive building
        filename (str): Filename for the json output
        data (str): String with json data to be written
    """
    file = Path.joinpath(serve_path, filename)
    with open(file, "w") as outfile:
        json.dump(data, outfile, indent=2)
        outfile.write("\n")
