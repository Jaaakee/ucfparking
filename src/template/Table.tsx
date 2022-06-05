import Link from "next/link";
import { Button } from "../button/Button";
import { DetailTable } from "../table/DetailTable";

const Table = () => (
  <DetailTable
    head={
      <tr>
        <th>Name</th>
        <th>Title</th>
        <th>Member since</th>
        <th>Status</th>
        <th>Action</th>
      </tr>
    }
    buttons={
      <>
        <Link href="/">
          <a>
            <Button sm secondary>
              Export
            </Button>
          </a>
        </Link>

        <Link href="/">
          <a>
            <Button sm>New User</Button>
          </a>
        </Link>
      </>
    }
    pagination={{
      stats: "1 - 10 of 350",
      current: 2,
      nbPage: 5,
      href: "/tables"
    }}
  >
    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>

    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>

    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>

    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>

    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>

    <tr>
      <td>John Doe</td>
      <td>Software Engineer</td>
      <td>12/09/2020</td>
      <td>Active</td>
      <td>
        <Link href="/">
          <a>Edit</a>
        </Link>
      </td>
    </tr>
  </DetailTable>
);

export { Table };
