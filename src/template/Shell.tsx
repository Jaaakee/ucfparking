import { ReactNode } from 'react';

import Link from 'next/link';

import { Button } from '../button/Button';
import { SidebarHeader } from '../shell/SidebarHeader';

type IShellProps = {
  title: string;
  children: ReactNode;
};

const Shell = (props: IShellProps) => (
  <SidebarHeader
    title={props.title}

    rightContent={
      <>
        <Link href="/">
          <a>
            <Button>API</Button>
          </a>
        </Link>
        <Link href="/">
          <a>
            <Button>GitHub</Button>
          </a>
        </Link>
      </>
    }
  >
    {props.children}
  </SidebarHeader>
);

export { Shell };
