import { ReactNode } from 'react';

type ILabelBoxButtonProps = {
  id: string;
  text: string;
  children: ReactNode;
};

const LabelBoxButton = (props: ILabelBoxButtonProps) => (
  <label htmlFor={props.id} className="label-box-button flex items-center">
    {props.children}

    <div className="ml-2">{props.text}</div>

    <style jsx>
      {`
        .label-box-button :global(input) {
          @apply border-gray-300 text-primary-600 shadow-sm hover:border-gray-300 focus:border-primary-300 focus:ring focus:ring-primary-200 focus:ring-opacity-50 focus:ring-offset-0;
        }

        .label-box-button :global(input[type='checkbox']) {
          @apply rounded;
        }
      `}
    </style>
  </label>
);

export { LabelBoxButton };
