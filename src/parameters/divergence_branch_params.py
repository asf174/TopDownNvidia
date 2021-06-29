"""
Class with all params of Divergence.Branch class
and their subclasses

@author:    Alvaro Saiz (UC)
@date:      Jan 2021
@version:   1.0
"""

class DivergenceBranchParameters:

    C_DIVERGENCE_BRANCH_NAME                    : str        = "DIVERGENCE BRANCH"
    C_DIVERGENCE_BRANCH_DESCRIPTION             : str        = ("It analyzes the parts of the GPU architecture where we have a loss of performance (IPC) due to\n"
                                                                + "memory bounds. This part takes into account aspects such as data dependencies, failures or access\n"
                                                                + "limits in caches")