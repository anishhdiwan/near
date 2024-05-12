import argparse
import os, yaml
from omegaconf import OmegaConf



if __name__ == "__main__":
    print("""

    This script generates .yaml files containing names and weights of motion files to train imitation learning algorithms. Run with the option --help for more info

    """)

    parser = argparse.ArgumentParser()                    
    parser.add_argument("-dir", "--dirname", type=str, required=True, help="name of the directory where motion data is stored")
    parser.add_argument("-n", "--name", type=str, default="motion_data", required=False, help="name of the motion dataset .yaml file")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dirname = os.path.join(current_dir, args.dirname)
    filename = args.name + ".yaml"
    assert os.path.exists(dirname), f"No such directory called {args.dirname} at {current_dir}"
    
    if os.path.exists(os.path.join(current_dir, filename)):
        tries = 0
        while tries < 5:
            overwrite = input(f"A file called {filename} already exists in this location. Are you sure you want to overwrite it? (y/n) ")
            if not isinstance(overwrite, str) or len(overwrite) > 1:
                print("please enter only single character y/n")
                tries += 1
                continue

            if overwrite == "n":
                quit()
            elif overwrite == "y":
                break
        
        if tries >=5:
            print("No valid inputs found!")
            quit()


    motions = []
    motion_files = [f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]   
    for idx, motion_entry in enumerate(motion_files):
        motions.append({'file' : f'{args.dirname}/{motion_entry}', 'weight' : 1.0})

    motion_data = OmegaConf.create()
    motion_data.motions = motions
    OmegaConf.save(config=motion_data, f=filename)



