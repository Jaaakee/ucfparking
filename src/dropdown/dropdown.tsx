import { Menu, Transition } from '@headlessui/react'
import { Fragment } from 'react'
import { ChevronDownIcon } from '@heroicons/react/solid'
import { useDispatch } from 'react-redux'
import { bindActionCreators } from 'redux'
import { actionCreators } from '../redux'

export default function Dropdown() {
  const dispatch = useDispatch()

  const { getLastDayData, getLastWeekData, getLastMonthData } =
    bindActionCreators(actionCreators, dispatch)
  return (
    <div className="">
      <Menu as="div" className="relative inline-block text-left">
        <div>
          <Menu.Button className="inline-flex w-full justify-center rounded-md text-center border text-lg font-semibold py-2 px-4 text-white bg-primary-500 dark:border-gray-800 border-gray-100  hover:bg-primary-600">
            Filters
            <ChevronDownIcon
              className="-mr-2 h-6 w-6 mt-0.5 text-primary-100 hover:text-primary-100 font-semibold"
              aria-hidden="true"
            />
          </Menu.Button>
        </div>
        <Transition
          as={Fragment}
          enter="transition ease-out duration-100"
          enterFrom="transform opacity-0 scale-95"
          enterTo="transform opacity-100 scale-100"
          leave="transition ease-in duration-75"
          leaveFrom="transform opacity-100 scale-100"
          leaveTo="transform opacity-0 scale-95"
        >
          <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right divide-y dark:bg-black dark:border dark:border-gray-800  divide-gray-100 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
            <div className="px-1 py-1">
              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active
                        ? 'bg-primary-500 text-white'
                        : 'dark:text-gray-400 text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-base font-semibold`}
                    onClick={() => getLastDayData()}
                  >
                    Past 24 Hours
                  </button>
                )}
              </Menu.Item>
              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active
                        ? 'bg-primary-500 text-white'
                        : 'dark:text-gray-400 text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-base font-semibold`}
                    onClick={() => {
                      getLastWeekData()
                    }}
                  >
                    Past Week Average
                  </button>
                )}
              </Menu.Item>

              <Menu.Item>
                {({ active }) => (
                  <button
                    className={`${
                      active
                        ? 'bg-primary-500 text-white'
                        : 'dark:text-gray-400 text-gray-900'
                    } group flex w-full items-center rounded-md px-2 py-2 text-base font-semibold`}
                    onClick={() => {
                      getLastMonthData()
                    }}
                  >
                    Past Month Average
                  </button>
                )}
              </Menu.Item>
            </div>
          </Menu.Items>
        </Transition>
      </Menu>
    </div>
  )
}
