"""
Program who filters the results obtained from topdown.py
and show the results.

@author Alvaro Saiz (UC)
@date May 2021
@version 1.0
"""

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(1, parentdir)
from parameters.parser_parameters import ParserParamters
from parameters.topdown_parameters import TopDownParameters
from pathlib import Path

class Parser:
    """ 
    Class which filters the results obtained from topdown.pt
    and show the results.
    """

    def __init__(self):
        """ 
        Init attributes depending of arguments.
        """

        self.__parser : argparse.ArgumentParse = argparse.ArgumentParser(
            formatter_class = lambda prog : argparse.HelpFormatter(prog, max_help_position = 50),
            description = "Parser results obtainded from topdown.py")
        self.__parser_optionals.title = "Optional Arguments"
        self.__add_arguments(self.__parser)

        # Save values
        args : argparse.Namespace = self.__parser.parse_args()
        
        self.__level : int = args.level
        self.__input_file : str = args.input_file
        #self.__verbose : bool = args.verbose
        pass

    def __add_input_file_argument(self, group : argparse._ArgumentGroup): 
        """ 
        Add input-file argument. 'C_INPUT_FILE_ARGUMENT_SHORT_OPTION' is the short option of argument
        and 'C_INPUT_FILE_ARGUMENT_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """
        
        group.add_argument(
            ParserParameters.C_INPUT_FILE_ARGUMENT_SHORT_OPTION,
            ParserParameters.C_INPUT_FILE_ARGUMENT_LONG_OPTION,
            required = True,
            help = TopDownParameters.C_INPUT_FILE_ARGUMENT_DESCRIPTION,
            default = None,
            nargs = '?',  
            action = DontRepeat,
            type = str,
            #metavar='/path/to/file',
            dest = 'input_file')
        pass
    
     def __add_level_argument(self, group : argparse._ArgumentGroup):
        """ 
        Add level argument. 'C_LEVEL_ARGUMENT_SHORT_OPTION' is the short option of argument
        and 'C_LEVEL_ARGUMENT_LONG_OPTION' is the long version of argument.

        Params:
            group : argparse._ArgumentGroup ; group of the argument.
        """
                                                      
        group.add_argument(
            TopDownParameters.C_LEVEL_ARGUMENT_SHORT_OPTION, TopDownParameters.C_LEVEL_ARGUMENT_LONG_OPTION,
            required = True,
            help = TopDownParameters.C_LEVEL_ARGUMENT_DESCRIPTION,
            type = int,
            action = DontRepeat,
            nargs = 1,
            default = -1, 
            choices = range(TopDownParameters.C_MIN_LEVEL_EXECUTION, TopDownParameters.C_MAX_LEVEL_EXECUTION + 1), # range [1,3], produces error, no if needed
            metavar = '[NUM]',
            dest = 'level')
        pass
    
    def __add_arguments(self, parser : argparse.ArgumentParser):
        """
        Add arguments of the pogram.
         
        Params:
            parser : argparse.ArgumentParser ; group of the arguments.
        """
        self.__add_level_argument(parser)
        self.__add_inputfile_argument(parser)
        
        pass

    def level(self) -> int:
        """
        Find the TopDown run level.
        
        Returns:
            the level of the execution.
        """
        
        return self.__level
        pass
    
    def input_file(self) -> str:
         """
         Find path to output file.

         Returns:
             path to input file.
         """

         return self.__input_file # descriptor to file or None
         pass

    def __is_nvprof_file(self):
        """ 
        Check if it's nvprof/nsight file.

        Returns:
            True if it's nvprof file, or False if it's nsight file
        """
        
        return False
        pass

    def launch(self):
        """Launch execution."""

        if self.__is_nvprof_file():

            front_end : FrontEndNvprof
            back_end : BackEndNvprof
            divergence : DivergenceNvprof
            retire : RetireNvprof
            extra_measure : ExtraMeasureNvprof
            if self.level() == 1:
                front_end = FrontEndNvprof(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NVPROF_L1_METRICS, FrontEndParameters.C_FRONT_END_NVPROF_L1_EVENTS)
                back_end = BackEndNvprof(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION, 
                    BackEndParameters.C_BACK_END_NVPROF_L1_METRICS, BackEndParameters.C_BACK_END_NVPROF_L1_EVENTS)
                divergence = DivergenceNvprof(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NVPROF_L1_METRICS, DivergenceParameters.C_DIVERGENCE_NVPROF_L1_EVENTS)
                retire = RetireNvprof(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NVPROF_L1_METRICS, RetireParameters.C_RETIRE_NVPROF_L1_EVENTS)
                extra_measure = ExtraMeasureNsight(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L1_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L1_EVENTS)
                level : LevelOneNvprof = LevelOneNvprof(program, self.input_file(), self.output_file(), show_metrics, show_events, front_end,
                    back_end, divergence, retire, extra_measure)
            elif self.level() == 2:
                front_end = FrontEndNvprof(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NVPROF_L2_METRICS, FrontEndParameters.C_FRONT_END_NVPROF_L2_EVENTS)
                back_end = BackEndNvprof(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION, 
                    BackEndParameters.C_BACK_END_NVPROF_L2_METRICS, BackEndParameters.C_BACK_END_NVPROF_L2_EVENTS)
                divergence = DivergenceNvprof(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NVPROF_L2_METRICS, DivergenceParameters.C_DIVERGENCE_NVPROF_L2_EVENTS)
                retire = RetireNvprof(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NVPROF_L2_METRICS, RetireParameters.C_RETIRE_NVPROF_L2_EVENTS)
                extra_measure = ExtraMeasureNvprof(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L2_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L2_EVENTS)
                front_band_width : FrontBandWidthNvprof = FrontBandWidthNvprof(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, 
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_L2_METRICS, 
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_L2_EVENTS)
                front_dependency : FrontDependencyNvprof = FrontDependencyNvprof(FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, 
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION, FrontDependencyParameters.C_FRONT_DEPENDENCY_NVPROF_L2_METRICS, 
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_NVPROF_L2_EVENTS)
                back_memory_bound : BackMemoryBoundNvprof = BackMemoryBoundNvprof(BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NAME, 
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION, BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_L2_METRICS, 
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_L2_EVENTS)
                back_core_bound : BackCoreBoundNvprof = BackCoreBoundNvprof(BackCoreBoundParameters.C_BACK_CORE_BOUND_NAME, 
                    BackCoreBoundParameters.C_BACK_CORE_BOUND_DESCRIPTION, BackCoreBoundParameters.C_BACK_CORE_BOUND_NVPROF_L2_METRICS, 
                    BackCoreBoundParameters.C_BACK_CORE_BOUND_NVPROF_L2_EVENTS)
                level : LevelTwoNvprof = LevelTwoNvprof(program, self.input_file(), self.output_file(), show_metrics, show_events, front_end, back_end,
                    divergence, retire, extra_measure, front_dependency, front_band_width, back_core_bound, back_memory_bound) 
            elif self.level() == 3:
                front_end = FrontEndNvprof(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NVPROF_L3_METRICS, FrontEndParameters.C_FRONT_END_NVPROF_L3_EVENTS)
                back_end = BackEndNvprof(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION, 
                    BackEndParameters.C_BACK_END_NVPROF_L3_METRICS, BackEndParameters.C_BACK_END_NVPROF_L3_EVENTS)
                divergence = DivergenceNvprof(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NVPROF_L3_METRICS, DivergenceParameters.C_DIVERGENCE_NVPROF_L3_EVENTS)
                retire = RetireNvprof(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NVPROF_L3_METRICS, RetireParameters.C_RETIRE_NVPROF_L3_EVENTS)
                extra_measure = ExtraMeasureNsight(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L3_METRICS, ExtraMeasureParameters.C_EXTRA_MEASURE_NVPROF_L3_EVENTS)
                front_band_width : FrontBandWidthNvprof = FrontBandWidthNvprof(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, 
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_L3_METRICS, 
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NVPROF_L3_EVENTS)
                front_dependency : FrontDependencyNvprof = FrontDependencyNvprof(FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, 
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION, FrontDependencyParameters.C_FRONT_DEPENDENCY_NVPROF_L3_METRICS, 
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_NVPROF_L3_EVENTS)
                back_memory_bound : BackMemoryBoundNvprof = BackMemoryBoundNvprof(BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NAME, 
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION, BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_L3_METRICS, 
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NVPROF_L3_EVENTS)
                back_core_bound : BackCoreBoundNvprof = BackCoreBoundNvprof(BackCoreBoundParameters.C_BACK_CORE_BOUND_NAME, 
                    BackCoreBoundParameters.C_BACK_CORE_BOUND_DESCRIPTION, BackCoreBoundParameters.C_BACK_CORE_BOUND_NVPROF_L3_METRICS, 
                    BackCoreBoundParameters.C_BACK_CORE_BOUND_NVPROF_L3_EVENTS)
                level : LevelThreeNvprof = LevelThreeNvprof(program, self.input_file(), self.output_file(), show_metrics, show_events, front_end, back_end,
                    divergence, retire, extra_meausre, front_dependency, front_band_width, back_core_bound, back_memory_bound)        
        else:
            front_end : FrontEndNsight
            back_end : BackEndNsight
            divergence : DivergenceNsight
            retire : RetireNsight
            extra_measure : ExtraMeasureNsight
            if self.level() == 1:
                front_end = FrontEndNsight(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NSIGHT_L1_METRICS)
                back_end = BackEndNsight(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION,
                    BackEndParameters.C_BACK_END_NSIGHT_L1_METRICS)
                divergence = DivergenceNsight(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NSIGHT_L1_METRICS)
                retire = RetireNsight(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NSIGHT_L1_METRICS)
                extra_measure = ExtraMeasureNsight(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_L1_METRICS)
                level : LevelOneNsight = LevelOneNsight(program, self.input_file(), self.output_file(), show_metrics, front_end, back_end, divergence, retire, extra_measure)
            elif self.level() == 2:
                front_end = FrontEndNsight(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NSIGHT_L2_METRICS)
                back_end = BackEndNsight(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION,
                    BackEndParameters.C_BACK_END_NSIGHT_L2_METRICS)
                divergence = DivergenceNsight(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NSIGHT_L2_METRICS)
                retire = RetireNsight(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NSIGHT_L2_METRICS)
                extra_measure = ExtraMeasureNsight(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_L2_METRICS)

                front_band_width : FrontBandWidthNsight = FrontBandWidthNsight(FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_L2_METRICS)
                
                front_dependency : FrontDependencyNsight = FrontDependencyNsight(FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION,
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_L2_METRICS)
                
                back_memory_bound : BackMemoryBoundNsight = BackMemoryBoundNsight(BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NAME, BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION,
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_L2_METRICS)
                
                back_core_bound : BackCoreBoundNsight = BackCoreBoundNsight(BackCoreBoundParameters.C_BACK_CORE_BOUND_NAME, BackCoreBoundParameters.C_BACK_CORE_BOUND_DESCRIPTION,
                front_band_width : FrontBandWidthNsight = (FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_L2_METRICS)
                
                front_dependency : FrontDependencyNsight = (FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION,
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_L2_METRICS)
                
                back_memory_bound : BackMemoryBoundNsight = (BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NAME, BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION,
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_L2_METRICS)
                
                back_core_bound : BackCoreBoundNsight = (BackCoreBoundParameters.C_BACK_CORE_BOUND_NAME, BackCoreBoundParameters.C_BACK_CORE_BOUND_DESCRIPTION,
                    BackCoreBoundParameters.C_BACK_CORE_BOUND_NSIGHT_L2_METRICS) 
                level : LevelTwoNsight = LevelTwoNsight(program, self.input_file(), self.output_file(), show_metrics, front_end, back_end, divergence, retire, extra_measure, front_band_width,
                    front_dependency, back_core_bound, back_memory_bound) 
            elif self.level() == 3:
                front_end = FrontEndNsight(FrontEndParameters.C_FRONT_END_NAME, FrontEndParameters.C_FRONT_END_DESCRIPTION,
                    FrontEndParameters.C_FRONT_END_NSIGHT_L3_METRICS)
                back_end = BackEndNsight(BackEndParameters.C_BACK_END_NAME, BackEndParameters.C_BACK_END_DESCRIPTION,
                    BackEndParameters.C_BACK_END_NSIGHT_L3_METRICS)
                divergence = DivergenceNsight(DivergenceParameters.C_DIVERGENCE_NAME, DivergenceParameters.C_DIVERGENCE_DESCRIPTION,
                    DivergenceParameters.C_DIVERGENCE_NSIGHT_L3_METRICS)
                retire = RetireNsight(RetireParameters.C_RETIRE_NAME, RetireParameters.C_RETIRE_DESCRIPTION,
                    RetireParameters.C_RETIRE_NSIGHT_L3_METRICS)
                extra_measure = ExtraMeasureNsight(ExtraMeasureParameters.C_EXTRA_MEASURE_NAME, ExtraMeasureParameters.C_EXTRA_MEASURE_DESCRIPTION,
                    ExtraMeasureParameters.C_EXTRA_MEASURE_NSIGHT_L3_METRICS)
                front_band_width : FrontBandWidthNsight = (FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NAME, FrontBandWidthParameters.C_FRONT_BAND_WIDTH_DESCRIPTION,
                    FrontBandWidthParameters.C_FRONT_BAND_WIDTH_NSIGHT_L3_METRICS)
                front_dependency : FrontDependencyNsight = (FrontDependencyParameters.C_FRONT_DEPENDENCY_NAME, FrontDependencyParameters.C_FRONT_DEPENDENCY_DESCRIPTION,
                    FrontDependencyParameters.C_FRONT_DEPENDENCY_NSIGHT_L3_METRICS)
                back_memory_bound : BackMemoryBoundNsight = (BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NAME, BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_DESCRIPTION,
                    BackMemoryBoundParameters.C_BACK_MEMORY_BOUND_NSIGHT_L3_METRICS)
                   back_core_bound : BackCoreBoundNsight = (BackCoreBoundParameters.C_BACK_CORE_BOUND_NAME, BackCoreBoundParameters.C_BACK_CORE_BOUND_DESCRIPTION,
                     BackCoreBoundParameters.C_BACK_CORE_BOUND_NSIGHT_L3_METRICS)
                level : LevelThreeNsight = LevelThreeNsight(program, self.input_file(), self.output_file(), show_metrics, front_end, back_end, divergence, retire, extra_measure, front_band_width,
                     front_dependency, back_core_bound, back_memory_bound)

                # set file content in str
                file_content : str = Path(self.input_file).read_text() 
                level.set_results(file_content)
        pass
