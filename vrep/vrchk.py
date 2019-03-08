from vrep.const import *
from warnings import warn

def vrchk(res):
    # Checks VREP return code. Set buffer to 1 if you are reading from a buffered call.
    
    # (C) Copyright Renaud Detry 2013.
    # Distributed under the GNU General Public License.
    # (See http://www.gnu.org/copyleft/gpl.html)
    
    if res == simx_error_noerror or res == simx_error_novalue_flag:
        return
    
    # if res == simx_error_novalue_flag:
    #     msg = 'There is no command reply in the input buffer. This should not always be ' \
    #           'considered as an error, depending on the selected operation mode'
    #     warn(msg)
    #     return 
    
    error_dict = {
        simx_error_timeout_flag: 
            'Thedef timed out (probably the network is down or too slow)',
        simx_error_illegal_opmode_flag:
            'The specified operation mode is not supported for the given function',
        simx_error_remote_error_flag:
            'Thedef caused an error on the server side (e.g. an invalid handle was specified)',
        simx_error_split_progress_flag:
            'The communication thread is still processing previous split command of the same type',
        simx_error_local_error_flag:
            'Thedef caused an error on the client side',
        simx_error_initialize_error_flag:
            'simxStart was not yet called',
    }
    expl = error_dict[res] if res in error_dict else 'Undefined error'
    
    raise Exception('Remote APIdef call returned with error code: %d.\nExplanation: %s.' % 
                    (res, expl))

