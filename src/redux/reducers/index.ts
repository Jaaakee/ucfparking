import { combineReducers } from "redux";
import ChartsReducer from "./ChartsReducer";
import parkingReducer from "./ParkingReducer";

const reducers = combineReducers({
  parking: parkingReducer,
  chart: ChartsReducer
});

export default reducers;

export type RootState = ReturnType<typeof reducers>;
