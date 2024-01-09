import sys
import getopt
import pandas as pd
from time import time
import pipeline

def usage():
    cmds = [
        'python all_in_one.py --in-mri <in_mri> --out_mri <out_mri> [options]',
        'python all_in_one.py --in-csv <in-csv> [options]'
    ]
    options = [
        ['--in-mri <input>','input MR image path, requires --out-mri'],
        ['--out-mri <output>','output MR image path, requires --in-mri'],
        ['--in-csv <csv>', 'CSV filepath with two columns: in_paths, out_paths. --in-mri must not be defined'],
        ['--hd-bet-cpu', 'runs HD-BET with cpu (GPU by default)'],
        ['--n-procs <n>', 'number of CPUs to use if several inputs (--in-csv option), default=1'],
        ['--help', 'displays this help']
    ]
    print('\n'.join(cmds))
    print('Available options are:')
    for opt,doc in options: print(f"\t{opt}{(20-len(opt))*' '}{doc}")

def main():
    try:
        opts,_ = getopt.getopt(sys.argv[1:], 'h', ['help', 'in-mri=', 'out-mri=', 'in-csv=', 'n-procs=', 'hd-bet-cpu'])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    
    in_mri = None
    out_mri = None
    in_csv = None
    n_procs = 1
    hd_bet_cpu = False
    for opt,arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        if opt == '--in-mri': in_mri = arg
        elif opt == '--out-mri': out_mri = arg
        elif opt == '--in-csv': in_csv = arg
        elif opt == '--hd-bet-cpu': hd_bet_cpu = True
        elif opt == '--n-procs': n_procs = int(arg)
    
    if (in_mri is None) ^ (out_mri is None) : 
        usage()
        sys.exit(2)
        
    if not ((in_mri is None) ^ (in_csv is None)):
        usage()
        sys.exit(2)
        
    if in_csv:
        df = pd.read_csv(in_csv)
        pipeline.run_multiproc(df.in_paths.tolist(), df.out_paths.tolist(), n_procs, hd_bet_cpu)
    else: pipeline.run_singleproc(in_mri, out_mri, hd_bet_cpu)
        
    
if __name__ == "__main__":
    t_start = time()
    main()
    print(f"End of execution, total processing time = {time()-t_start:.0f} seconds")

