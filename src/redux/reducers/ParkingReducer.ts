import { Action } from "../actions";
import { ActionType } from "../constants";

const initialState = {};

// eslint-disable-next-line @typescript-eslint/default-param-last
const reducer = (state = initialState, action: Action): Object => {
  switch (action.type) {
    case ActionType.LAST_ROW:
      return action.payload;

    default:
      return state;
  }
};

export default reducer;
