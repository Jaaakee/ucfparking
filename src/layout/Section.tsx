import { ReactNode } from "react";

type ISection = {
  children: ReactNode;
};

const Section = (props: ISection) => (
  <div className="px-3 sm:px-5 lg:px-6 my-6">{props.children}</div>
);

export { Section };
