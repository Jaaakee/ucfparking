import { NextSeo } from "next-seo";
import Head from "next/head";

import { AppConfig } from "../utils/AppConfig";

type IMetaProps = {
  title: string;
  description: string;
  canonical?: string;
};

const Meta = (props: IMetaProps) => (
  <>
    <Head>
      <meta charSet="UTF-8" key="charset" />
      <meta
        name="viewport"
        content="width=device-width,initial-scale=1"
        key="viewport"
      />

      <link rel="apple-touch-icon" sizes="180x180" href="dark/apple-touch-icon.png" />
      <link rel="icon" type="image/png" sizes="32x32" href="dark/favicon-32x32.png" />
      <link rel="icon" type="image/png" sizes="16x16" href="dark/favicon-16x16.png" />
      <link rel="manifest" href="dark/site.webmanifest" />
    </Head>
    <NextSeo
      title={props.title}
      description={props.description}
      canonical={props.canonical}
      openGraph={{
        title: props.title,
        description: props.description,
        url: props.canonical,
        locale: AppConfig.locale,
        site_name: AppConfig.site_name
      }}
    />
  </>
);

export { Meta };
