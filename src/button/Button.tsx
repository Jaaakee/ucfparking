import className from 'classnames'

type IButtonProps = {
  xs?: boolean
  sm?: boolean
  xl?: boolean
  secondary?: boolean
  full?: boolean
  children: string
}

const Button = (props: IButtonProps) => {
  const btnClass = className({
    btn: true,
    'btn-xs': props.xs,
    'btn-sm': props.sm,
    'btn-xl': props.xl,
    'btn-base': !props.xl,
    'btn-secondary': props.secondary,
    'btn-primary': !props.secondary,
    'w-full': props.full
  })

  return (
    <button className={btnClass}>
      {props.children}

      <style jsx>
        {`
          .btn {
            @apply inline-block rounded-md text-center border;
          }

          .btn-base {
            @apply text-lg font-semibold py-2 px-4;
          }

          .btn-xl {
            @apply font-extrabold text-xl py-4 px-6;
          }

          .btn-xs {
            @apply text-sm py-2 px-2;
          }

          .btn-sm {
            @apply text-base font-medium py-2 px-3;
          }

          .btn-primary {
            @apply text-white bg-primary-500 dark:border-gray-800 border-gray-100;
          }

          .btn-primary:hover {
            @apply bg-primary-600;
          }

          .btn-primary:active {
            @apply bg-primary-500;
          }

          .btn-secondary {
            @apply bg-white text-primary-500 border-gray-200;
          }

          .btn-secondary:hover {
            @apply bg-gray-100;
          }

          .btn-secondary:active {
            @apply border-gray-200;
          }
        `}
      </style>
    </button>
  )
}

export { Button }
