import { ReactNode } from "react";

type ILabelInputProps = {
  text: string;
  children: ReactNode;
};

const Label = (props: ILabelInputProps) => (
  <div className="my-6 flex flex-col sm:flex-row sm:items-center">
    <div className="sm:w-2/6 xl:w-1/6 text-gray-700">{props.text}</div>

    <div className="mt-1 sm:mt-0 sm:ml-3 flex flex-row space-x-5">
      {props.children}
    </div>
  </div>
);

export { Label };
