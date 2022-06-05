import { ActionType } from "../constants";

interface LastRowAction {
  type: ActionType.LAST_ROW;
  payload: number;
}

interface HandleAllData {
  type: 
    // ActionType.GET_TODAY_DATA
    | ActionType.GET_LAST_DAY_DATA
    | ActionType.GET_LAST_WEEK_DATA
    | ActionType.GET_LAST_MONTH_DATA
    // | ActionType.GET_LAST_YEAR_DATA
    // ActionType.GET_ALL_DATA |
    | ActionType.GET_BAR_CHART_DATA
    | ActionType.GET_PIE_CHART_DATA
     ticks: any;
    payload: any;
    data: any;
}

export type Action = LastRowAction | HandleAllData;
