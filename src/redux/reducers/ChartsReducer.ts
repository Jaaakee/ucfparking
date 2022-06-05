import { Action } from "../actions";
import { ActionType } from "../constants";

const initialState = {
  lineChart: [],
  lineChartData: [],
  lineChartTicks: [],

  barChart: [],
  barChartData: [],

  pieChart: [],
  pieChartData: []
};

// eslint-disable-next-line @typescript-eslint/default-param-last
const chartReducer = (state: any = initialState, action: Action): any => {
  switch (action.type) {
    // case ActionType.GET_TODAY_DATA:
    //     return {
    //         ...state,
    //         lineChart: action.payload,
    //         lineChartData: action.data,
    //     }
    case ActionType.GET_LAST_DAY_DATA:
      return {
        ...state,
        lineChart: action.payload,
        lineChartData: action.data,
        lineChartTicks: action.ticks

      };
    case ActionType.GET_LAST_WEEK_DATA:
      return {
        ...state,
        lineChart: action.payload,
        lineChartData: action.data,
        lineChartTicks: action.ticks
      };
    case ActionType.GET_LAST_MONTH_DATA:
      return {
        ...state,
        lineChart: action.payload,
        lineChartData: action.data,
        lineChartTicks: action.ticks
      };
    // case ActionType.GET_LAST_YEAR_DATA:
    //   return {
    //     ...state,
    //     lineChart: action.payload,
    //     lineChartData: action.data
    //   };
    // case ActionType.GET_ALL_DATA:
    //     return {
    //         ...state,
    //         lineChart: action.payload
    //     }
    case ActionType.GET_BAR_CHART_DATA:
      return {
        ...state,
        barChart: action.payload,
        barChartData: action.data
      };
    case ActionType.GET_PIE_CHART_DATA:
      return {
        ...state,
        pieChart: action.payload,
        pieChartData: action.data
      };
    default:
      return state;
  }
};

export default chartReducer;
