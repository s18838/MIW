'''
This is a temporary script file.
'''
import argparse




def main(args):
    '''
    This is main function of this module, it prints hello world and value of demo arguments

    Parameters
    ----------
    args : argparse.Namespace
        Namespace containing all of argument from command line.

    Returns
    -------
    None.

    '''
    print(f'Type is {type(args)}.')
    print('Hello world but from main function.')
    print(f'Value of demo argument is  {args.demo_argument}.')

def parse_arguments():
    '''
    This function parses arguments from command line

    Returns
    -------
    argparse.Namespace
        Namespace containing all of argument from command line or their default values.

    '''
    parser = argparse.ArgumentParser(description=('This is simple hello world'))
    parser.add_argument('-d',
                        '--demo_argument',
                        type=str,
                        default='defaul value',
                        help='This is just a demo argument')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())
