import { AppProps } from "next/app";
import { Provider } from "react-redux";
import { store } from "../redux/store";

import "../styles/global.css";

const MyApp = ({ Component, pageProps }: AppProps | any) => (
  <>
    <Provider store={store}>
      <Component {...pageProps} />
    </Provider>
  </>
);

export default MyApp;
