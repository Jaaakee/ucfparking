import { Dispatch } from "redux";
import { ActionType } from "../constants/index";
import { filterByWeek, handelChartData, handleBarChart } from "./chartActions";

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
