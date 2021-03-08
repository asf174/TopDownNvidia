def _get_percentage_time(i):
    return float(50)
def _get_total_value_of_list(list_values : list[str]) -> float:
        """
        Get total value of list of metric/event
    
        Params:
            list_values : list[str] ; list to be computed
        """

        i : int = 0
        total_value : float = 0.0
        for value in list_values:
            if value[len(value) - 1] == "%":
                total_value += float(value[0:len(value) - 1])*(_get_percentage_time(i)/100.0)
            else:
                total_value += float(value)*(_get_percentage_time(i)/100.0)
            i += 1
        return total_value
        pass

print(_get_total_value_of_list(["2","1"]))