class LevelTwoNsight:

    def _get_results(self, lst_output : list):
        """ 
        Get results of the different parts.

        Parameters:
            lst_output              : list     ; OUTPUT list with results
        """

        # revisar en unos usa atributo y en otros la llamada al metodo
        #  Keep Results
        converter : MessageFormat = MessageFormat()
        if not self._recolect_metrics:
            return
        if self._recolect_metrics and self._front_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_end.name()))
            super()._add_result_part_to_lst(self._front_end.metrics(), 
                self._front_end.metrics_description(), lst_output, True)
        if  self._recolect_metrics and self._front_band_width.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_band_width.name()))
            super()._add_result_part_to_lst(self._front_band_width.metrics(), 
                self._front_band_width.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._front_dependency.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._front_dependency.name()))
            super()._add_result_part_to_lst(self._front_dependency.metrics(), 
                self._front_dependency.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_end.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_end.name()))
            super()._add_result_part_to_lst(self._back_end.metrics(), 
                self._back_end.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_core_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_core_bound.name()))
            super()._add_result_part_to_lst(self._back_core_bound.metrics(), 
                self._back_core_bound.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._back_memory_bound.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._back_memory_bound.name()))
            super()._add_result_part_to_lst(self._back_memory_bound.metrics(), 
                self._back_memory_bound.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._divergence.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._divergence.name()))
            super()._add_result_part_to_lst(self._divergence.metrics(), 
                self._divergence.metrics_description(), lst_output, True)
        if self._recolect_metrics and  self._retire.metrics_str() != "":
                lst_output.append(converter.underlined_str(self._retire.name()))
                super()._add_result_part_to_lst(self._retire.metrics(), 
                self._retire.metrics_description(), lst_output, True)
        if self._recolect_metrics and self._extra_measure.metrics_str() != "":
            lst_output.append(converter.underlined_str(self._extra_measure.name()))
            super()._add_result_part_to_lst(self._extra_measure.metrics(), 
                self._extra_measure.metrics_description(), lst_output, True)
        lst_output.append("\n")
        pass

    pass