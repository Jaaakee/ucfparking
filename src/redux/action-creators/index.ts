import { Dispatch } from "redux";
import { ActionType } from "../constants/index";
import { filterByWeek, handelChartData, handleBarChart } from "./chartActions";

// export const HandleLastRowData = (lastRowData: any) => (dispatch: Dispatch) => {
export const HandleLastRowData = () => async(dispatch: Dispatch) => {
  try {
    const res = await fetch("https://api.ucfparking.com/stats");
  const lastRowData = await res.json();
  dispatch({
    type: ActionType.LAST_ROW,
    payload: lastRowData
  });
  } catch (error) {
    console.log(error)
  }
};

// export const getTodayData = () => async (dispatch: Dispatch) => {
// try {
//     const res = await fetch("https://api.ucfparking.com/today");
//     const graphData= await res.json();

//     let chartData = handelChartData(graphData.data, "today")
//     // console.log("handelChartData", graphData.data)
//     dispatch({
//         type: ActionType.GET_TODAY_DATA,
//         payload: graphData.data,
//         data: chartData
//     });
// } catch (error) {
  // console.log(error)
// }
//   };
export const getLastDayData = () => async (dispatch: Dispatch) => {
  try {
    const res = await fetch("https://api.ucfparking.com/lastday");
  const graphData = await res.json();
  const chartData = handelChartData(graphData.data, "lastday");
  dispatch({
    type: ActionType.GET_LAST_DAY_DATA,
    payload: graphData.data,
    data: chartData.chartData,
    ticks: chartData.lineGraphTicks
  });
  } catch (error) {
    console.log(error)
  }
};

export const getLastWeekData = () => async (dispatch: Dispatch) => {
  try {
    const res = await fetch("https://api.ucfparking.com/week");
  const graphData = await res.json();
  const chartData = filterByWeek(graphData.data);
  dispatch({
    type: ActionType.GET_LAST_WEEK_DATA,
    payload: graphData.data,
    data: chartData.chartData,
    ticks: chartData.lineGraphTicks
  });
  } catch (error) {
    console.log(error)
  }
};

export const getLastMonthData = () => async (dispatch: Dispatch) => {
  try {
  const res = await fetch("https://api.ucfparking.com/lastmonth");
  const graphData = await res.json();  
  const chartData = filterByWeek(graphData.data);
  dispatch({
    type: ActionType.GET_LAST_MONTH_DATA,
    payload: graphData.data,
    data: chartData.chartData,
    ticks: chartData.lineGraphTicks
  });
  } catch (error) {
    console.log(error)
  }
};

// export const getLastYearData = () => async (dispatch: Dispatch) => {
//   const res = await fetch("https://api.ucfparking.com/lastyear");
//   const graphData = await res.json();
//   const chartData = handelChartData(graphData.data, "lastyear");
//   dispatch({
//     type: ActionType.GET_LAST_YEAR_DATA,
//     payload: graphData.data,
//     data: chartData
//   });
// };

//   export const handelAllData = () => async (dispatch: Dispatch) => {
//     const res = await fetch("https://api.ucfparking.com/all");
//     const graphData= await res.json();

//     dispatch({
//         type: ActionType.GET_ALL_DATA,
//         payload: graphData.data
//     });
//   };

export const getBarChartData = () => async (dispatch: Dispatch) => {
  try {
  const res = await fetch("https://api.ucfparking.com");
  const graphData = await res.json();
  const data = graphData.garages;
  const chartData = handleBarChart(data, "barchart");
  dispatch({
    type: ActionType.GET_BAR_CHART_DATA,
    payload: graphData.garages,
    data: chartData
  });
  } catch (error) {
    console.log(error)
  }
};

export const getPieChartData = () => async (dispatch: Dispatch) => {
  try {
    const res = await fetch("https://api.ucfparking.com");
  const graphData = await res.json();
  const data = graphData.garages;
  const chartData = handleBarChart(data, "piechart");
  dispatch({
    type: ActionType.GET_PIE_CHART_DATA,
    payload: graphData.garages,
    data: chartData
  });
} catch (error) {
    console.log(error)    
  }
};
