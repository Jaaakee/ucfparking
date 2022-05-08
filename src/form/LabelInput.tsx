import { ReactNode } from 'react';

import classNames from 'classnames';

type ILabelInputProps = {
  id?: string;
  text: string;
  children: ReactNode;
  sm?: boolean;
};

const LabelInput = (props: ILabelInputProps) => {
  const inputBlockClass = classNames({
    'sm:w-4/6': !props.sm,
    'sm:w-3/6': props.sm,
    'xl:w-3/6': !props.sm,
    'xl:w-2/6': props.sm,
    'sm:ml-3': true,
    'mt-1': true,
    'sm:mt-0': true,
  });

  return (
    <label
      htmlFor={props.id}
      className="label-input my-6 flex flex-col sm:flex-row sm:items-center"
    >
      <div className="sm:w-2/6 xl:w-1/6 text-gray-700">{props.text}</div>

      <div className={inputBlockClass}>{props.children}</div>

      <style jsx>
        {`
          .label-input :global(input),
          .label-input :global(select),
          .label-input :global(textarea) {
            @apply w-full rounded-md border-gray-300 shadow-sm focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50;
          }
        `}
      </style>
    </label>
  );
};

export { LabelInput };
